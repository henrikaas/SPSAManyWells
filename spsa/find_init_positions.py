#!/usr/bin/env python3
"""
Walk each well's decision vector toward a target (choke=0.5, gas lift=1.0) and
save the simulation results at each iteration.

- Starts from the current well positions
- Moves choke by 0.025 and gas lift by 0.125 per iteration toward the target
- Saves data for every iteration
- If a simulation fails, uses the last valid data point for that well
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
    
    guesses = well.x_guesses
    for i in range(0, len(guesses)):
        try:
            simulator.x_guess = guesses[i]
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
        well.x_guesses.append(x)
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
        return well_idx, "ok", dp, x
    except SimError:
        return well_idx, "fail", None, None

def _run_iteration(
    wells: list[Well],
    well_data: list[pd.DataFrame],
    last_valid: list[Optional[pd.DataFrame]],
    sim_type: str,
    pool: mp.Pool,
) -> None:
    tasks = [(idx, well, sim_type) for idx, well in enumerate(wells)]
    results = pool.starmap(_parallel_step, tasks)

    for idx, status, dp, x in results:
        if status in ("ok", "choked"):
            if x is not None:
                wells[idx].x_guesses.append(x)
            well_data[idx] = pd.concat([well_data[idx], dp], ignore_index=True)
            last_valid[idx] = dp
        elif last_valid[idx] is not None:
            dp = last_valid[idx].copy(deep=True)
            well_data[idx] = pd.concat([well_data[idx], dp], ignore_index=True)

def walk_init_positions(
    wells: list[Well],
    save_path: str,
    *,
    target_choke: float = 0.5,
    target_gas_lift: float = 1.0,
    step_choke: float = 0.025,
    step_gas_lift: float = 0.125,
) -> None:
    well_data = [create_sim_results_df() for _ in wells]
    last_valid = [None for _ in wells]

    def steps_needed(well: Well) -> int:
        u_steps = int(np.ceil(abs(target_choke - well.bc.u) / step_choke))
        if well.has_gas_lift:
            gl_steps = int(np.ceil(abs(target_gas_lift - well.bc.w_lg) / step_gas_lift))
        else:
            gl_steps = 0
        return max(u_steps, gl_steps)

    total_iters = max(steps_needed(w) for w in wells)

    with mp.Pool(processes=min(len(wells), max(1, mp.cpu_count() - 1))) as pool:
        _run_iteration(wells, well_data, last_valid, "Init walk", pool)
        save_data(wells, well_data=well_data, main_path=save_path, k=0)

        for k in range(1, total_iters + 1):
            for well in wells:
                well.bc.u = _step_towards(well.bc.u, target_choke, step_choke, lower=0.0, upper=1.0)
                if well.has_gas_lift:
                    well.bc.w_lg = _step_towards(well.bc.w_lg, target_gas_lift, step_gas_lift, lower=0.0, upper=5.0)

            _run_iteration(wells, well_data, last_valid, "Init walk", pool)
            save_data(wells, well_data=well_data, main_path=save_path, k=k)


if __name__ == "__main__":
    n_runs = 1
    experiment = [{"config": "nsol_set2",
        "save": "init_positions",
        }]
    work_dir, results_dir = create_dirs(experiment, n_runs)

    wells = configure_wells(filepath=
                                    os.path.join(work_dir, "config files", f"{experiment[0]['config']}.csv"))

    walk_init_positions(wells,
                        save_path=os.path.join(results_dir, experiment[0]['save'], f"run{0}"))
