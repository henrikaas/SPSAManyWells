#!/usr/bin/env python3
"""
Walk each well's decision vector toward a target (choke=0.5, gas lift=1.0) and
save the simulation results at each iteration.

- Starts from the current well positions
- Moves choke by 0.025 and gas lift by 0.125 per iteration toward the target
- Saves data for every iteration
- If a simulation fails, retries with gas lift held constant; otherwise stops simulating that well
- Optional random-walk mode moves in random directions for a fixed number of iterations
- Optional combined gas-lift constraint rescales well gas lift values to keep the total under a limit
"""

from __future__ import annotations
import os
import multiprocessing as mp
from typing import Optional

import numpy as np
import pandas as pd

from spsa.utils import (
    configure_wells,           # preferred CSV->Well loader
    create_sim_results_df,     # preferred empty DF factory
    create_data_point,         # preferred row builder
    choked_flow,               # preferred choked-flow helper
    create_dirs,               # (optional) to mirror folder structure
    save_data,                 # (optional) if you want SPSA-style saving
)
 # Simulator stack
from manywells.simulator import SSDFSimulator, SimError
from scripts.data_generation.well import Well  # project Well object


# -------------------------
# Helpers
# -------------------------

def _single_simulation(simulator: SSDFSimulator, well: Well):

    #TODO: Analyze if this is needed. If so, handle more elegantly
    # # Skip simulation if choke is nearly closed
    # if simulator.bc.u <= 0.05:
    #     return None
    # if simulator.bc.u == 0.0:
    #     x = handle_choked_flow(well)
    #     return x
    try:
        x = simulator.simulate()
        return x
    except SimError as e:
        print(f"Simulation failed: {e}. Trying guesses...")
    
    guesses = well.x_guesses.x0_candidates
    for i in range(0, len(guesses)):
        try:
            simulator.x_guess = guesses[i]["x_guess"]
            x = simulator.simulate()
            simulator.x_guess = None # Reset the guess if successful simulation
            return x
        
        except SimError as e:
            print(f"Simulation failed: {e}. Trying next guess...")
            continue

    raise SimError(f"Could not simulate well after {len(guesses)} attempts. No guesses left.")

def _step_towards(value: float, target: float, step: float, *, lower: float, upper: float) -> float:
    if np.isclose(value, target):
        return float(np.clip(value, lower, upper))
    if value < target:
        next_value = min(value + step, target)
    else:
        next_value = max(value - step, target)
    return float(np.clip(next_value, lower, upper))

def _step_random(value: float, step: float, *, lower: float, upper: float, rng: np.random.Generator) -> float:
    direction = 1.0 if rng.integers(0, 2) == 1 else -1.0
    next_value = value + direction * step
    return float(np.clip(next_value, lower, upper))

def _enforce_combined_gas_lift(
    wells: list[Well],
    combined_gas_lift_max: float,
    active: Optional[list[bool]] = None,
) -> None:
    if combined_gas_lift_max < 0:
        raise ValueError("combined_gas_lift_max must be >= 0.")
    if active is None:
        active = [True for _ in wells]
    gas_lift_indices = [idx for idx, well in enumerate(wells) if well.has_gas_lift]
    if not gas_lift_indices:
        return
    inactive_gas_lift = sum(
        wells[idx].bc.w_lg for idx in gas_lift_indices if not active[idx]
    )
    active_indices = [idx for idx in gas_lift_indices if active[idx]]
    if not active_indices:
        return
    allowed_gas_lift = combined_gas_lift_max - inactive_gas_lift
    if allowed_gas_lift <= 0.0:
        for idx in active_indices:
            wells[idx].bc.w_lg = 0.0
        return
    total_active = sum(wells[idx].bc.w_lg for idx in active_indices)
    if total_active <= allowed_gas_lift or total_active == 0.0:
        return
    scale = allowed_gas_lift / total_active
    for idx in active_indices:
        wells[idx].bc.w_lg *= scale

def _simulate_well(
    well: Well,
    sim: SSDFSimulator,
    sim_type: str,
    last_valid_dp: Optional[pd.DataFrame],
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if well.bc.u <= 0.025:
        dp, _ = choked_flow(well, sim_type)
        return dp, dp

    try:
        x = _single_simulation(simulator=sim, well=well)
        well.x_guesses.add_candidate(x, sim, well)
        dp = create_data_point(well=well, sim=sim, x=x, sim_type=sim_type)
        return dp, dp
    except SimError:
        if last_valid_dp is None:
            return None, None
        return last_valid_dp.copy(deep=True), last_valid_dp

def _parallel_step(well_idx: int, well: Well, sim_type: str) -> tuple[int, str, Optional[pd.DataFrame], Optional[np.ndarray]]:
    sim = SSDFSimulator(well.wp, well.bc)
    if well.bc.u <= 0.025:
        dp, _ = choked_flow(well, sim_type)
        return well_idx, "choked", dp, None

    try:
        x = _single_simulation(simulator=sim, well=well)
        dp = create_data_point(well=well, sim=sim, x=x, sim_type=sim_type)
        well.x_guesses.add_candidate(x, sim, well)
        return well_idx, "ok", dp, x
    except SimError:
        return well_idx, "fail", None, None

def _retry_simulation(well: Well, sim_type: str) -> tuple[str, Optional[pd.DataFrame]]:
    sim = SSDFSimulator(well.wp, well.bc)
    if well.bc.u <= 0.025:
        dp, _ = choked_flow(well, sim_type)
        return "choked", dp

    try:
        x = _single_simulation(simulator=sim, well=well)
        dp = create_data_point(well=well, sim=sim, x=x, sim_type=sim_type)
        well.x_guesses.add_candidate(x, sim, well)
        return "ok", dp
    except SimError:
        return "fail", None

def _run_iteration(
    wells: list[Well],
    well_data: list[pd.DataFrame],
    last_valid: list[Optional[pd.DataFrame]],
    sim_type: str,
    pool: mp.Pool,
    active: list[bool],
) -> dict[int, str]:
    tasks = [(idx, wells[idx], sim_type) for idx, is_active in enumerate(active) if is_active]
    results = pool.starmap(_parallel_step, tasks)

    statuses: dict[int, str] = {}
    for idx, status, dp, x in results:
        statuses[idx] = status
        if status in ("ok", "choked"):
            well_data[idx] = pd.concat([well_data[idx], dp], ignore_index=True)
            last_valid[idx] = dp

    return statuses

def walk_init_positions(
    wells: list[Well],
    save_path: str,
    *,
    target_choke: float = 0.5,
    target_gas_lift: float = 1.0,
    step_choke: float = 0.025,
    step_gas_lift: float = 0.1,
    random_walk: bool = False,
    random_iters: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    combined_gas_lift_max: Optional[float] = 40,
) -> None:
    well_data = [create_sim_results_df() for _ in wells]
    last_valid = [None for _ in wells]
    active = [True for _ in wells]
    prev_states: list[tuple[float, float]] = [(well.bc.u, well.bc.w_lg) for well in wells]

    if random_walk:
        if random_iters is None or random_iters < 1:
            raise ValueError("random_iters must be >= 1 when random_walk is enabled.")
        total_iters = random_iters
        rng = rng or np.random.default_rng()
    else:
        def steps_needed(well: Well) -> int:
            u_steps = int(np.ceil(abs(target_choke - well.bc.u) / step_choke))
            if well.has_gas_lift:
                gl_steps = int(np.ceil(abs(target_gas_lift - well.bc.w_lg) / step_gas_lift))
            else:
                gl_steps = 0
            return max(u_steps, gl_steps)

        total_iters = max(steps_needed(w) for w in wells)

    with mp.Pool(processes=min(len(wells), max(1, mp.cpu_count() - 1))) as pool:
        if combined_gas_lift_max is not None:
            _enforce_combined_gas_lift(wells, combined_gas_lift_max, active)
        statuses = _run_iteration(wells, well_data, last_valid, "Init walk", pool, active)
        for idx, status in statuses.items():
            if status == "fail":
                active[idx] = False
        save_data(wells, well_data=well_data, main_path=save_path, k=0)

        for k in range(1, total_iters + 1):
            for idx, well in enumerate(wells):
                if not active[idx]:
                    continue
                prev_states[idx] = (well.bc.u, well.bc.w_lg)
                if random_walk:
                    well.bc.u = _step_random(well.bc.u, step_choke, lower=0.25, upper=1.0, rng=rng)
                    if well.has_gas_lift:
                        well.bc.w_lg = _step_random(well.bc.w_lg, step_gas_lift, lower=0.25, upper=5.0, rng=rng)
                else:
                    well.bc.u = _step_towards(well.bc.u, target_choke, step_choke, lower=0.0, upper=1.0)
                    if well.has_gas_lift:
                        well.bc.w_lg = _step_towards(well.bc.w_lg, target_gas_lift, step_gas_lift, lower=0.0, upper=5.0)

            if combined_gas_lift_max is not None:
                _enforce_combined_gas_lift(wells, combined_gas_lift_max, active)

            statuses = _run_iteration(wells, well_data, last_valid, "Init walk", pool, active)
            for idx, status in statuses.items():
                if status != "fail":
                    continue
                if not active[idx]:
                    continue
                prev_u, prev_w_lg = prev_states[idx]
                well = wells[idx]
                if well.has_gas_lift:
                    desired_w_lg = prev_w_lg
                    if combined_gas_lift_max is not None:
                        total_other = sum(
                            w.bc.w_lg
                            for j, w in enumerate(wells)
                            if j != idx and w.has_gas_lift
                        )
                        allowed = max(0.0, combined_gas_lift_max - total_other)
                        desired_w_lg = min(desired_w_lg, allowed)
                    well.bc.w_lg = float(np.clip(desired_w_lg, 0.0, 5.0))
                retry_status, dp = _retry_simulation(well, "Init walk")
                if retry_status in ("ok", "choked"):
                    well_data[idx] = pd.concat([well_data[idx], dp], ignore_index=True)
                    last_valid[idx] = dp
                else:
                    well.bc.u = prev_u
                    if well.has_gas_lift:
                        well.bc.w_lg = prev_w_lg
                    if last_valid[idx] is not None:
                        dp = last_valid[idx].copy(deep=True)
                        well_data[idx] = pd.concat([well_data[idx], dp], ignore_index=True)
                    active[idx] = False
        save_data(wells, well_data=well_data, main_path=save_path, k=k)


if __name__ == "__main__":
    n_runs = 1
    experiment = [{"config": "nsol_32wells_choke50-new",
        "save": "nsol_32wells_random",
        }]
    work_dir, results_dir = create_dirs(experiment, n_runs)

    wells = configure_wells(filepath=
                                    os.path.join(work_dir, "config files", f"{experiment[0]['config']}.csv"))

    walk_init_positions(wells,
                        save_path=os.path.join(results_dir, experiment[0]['save'], f"run{0}"),
                        random_walk=True,
                        random_iters=100)
