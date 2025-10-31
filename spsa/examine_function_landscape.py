#!/usr/bin/env python3
"""
Evaluate a 2D (choke, gas lift) simulation landscape per well and save results.

- Reads wells from a CSV (same mechanism as the SPSA script via `configure_wells` when available)
- Spawns a multiprocessing pool where **each task is one well**
- Evaluates a 2D grid of 50x50 combinations over:
    choke in [0, 1], gas lift in [0, 5]
- If choke == 0.0, returns zero flow (same behavior as SPSA code's choked-flow handling)
- Saves per-well data to disk in a "well_data" file per well, and an all-wells aggregate

This file is designed to run inside the same repository as the SPSA optimizer.
If project utilities are available (spsa.utils, manywells.simulator, etc.), the
script will use them. Otherwise, it will still produce CSVs with a generic schema.
"""

from __future__ import annotations
import argparse
import os
import sys
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Iterable

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
# Worker
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

def _parallel_simulation(well_idx: int, well: Well, sim: SSDFSimulator):
    """
    Simulates a well to a well and run a simulation.

    Args:
        well_idx (int): Index of the well in the system.
        well (Well): The well object to be perturbed and simulated.
        sim (SSDFSimulator): The simulator object to use for the simulation.

    Returns:
        tuple[int, np.ndarray | None]: A tuple containing the well index and the simulation result.
                                        If the simulation fails, the result is None.
    """
    if well.bc.u <= 0.025:
        x = [0.0] * ((sim.n_cells + 1) * sim.dim_x) # All zeros if choke is fully closed
        return well_idx, x
    try:
        x = _single_simulation(simulator=sim, well=well)
    except SimError:
        return well_idx, None

    return well_idx, x

def evaluate_well_grid(well_idx: int, well: Well, n_points: int = 50) -> pd.DataFrame:
    """Evaluate 2D grid for one well.

    Each axis has `n_points` points: choke in [0, 1], gas lift in [0, 5].
    If choke == 0.0 -> zero flow.
    """
    # Build grid
    choke_vals = np.linspace(0.0, 1.0, n_points)
    gl_vals = np.linspace(0.0, 5.0, n_points)

    # Prepare simulator (project path) if available
    sim = SSDFSimulator(well.wp, well.bc)

    data = create_sim_results_df()
    for c in choke_vals:
        for g in gl_vals:
            well.bc.u = c
            well.bc.w_lg = g

            _, x = _parallel_simulation(well_idx, well, sim)
            if x is not None:
                well.x_guesses.append(x)
                dp = create_data_point(well=well, sim=sim, x=x, sim_type='Grid eval')
                data = pd.concat([data, dp], ignore_index=True)

    return well_idx, well, data


def evaluate_function_landscape(wells: list[Well], save_path: str, n_points: int = 50) -> None:
    tasks = [(idx, w, n_points) for idx, w in enumerate(wells)]

    with mp.Pool(processes=min(len(wells), max(1, mp.cpu_count() - 1))) as pool:
        results = pool.starmap(evaluate_well_grid, tasks)

    well_data = [create_sim_results_df() for _ in wells]
    for idx, well, df in results:
        # Do something with the DataFrame `df` if needed
        well_data[idx] = pd.concat([well_data[idx], df], ignore_index=True)

    save_data(wells, well_data=well_data, main_path=save_path, k=0)


if __name__ == "__main__":
    grid_size = 50
    n_runs = 1
    experiment = [{"config": "mixedprod_choke50",
        "save": "grid evaluation",
        }]
    work_dir, results_dir = create_dirs(experiment, n_runs)

    wells = configure_wells(filepath=
                                    os.path.join(work_dir, "config files", f"{experiment[0]['config']}.csv"))

    evaluate_function_landscape(wells, n_points=grid_size,
                                save_path=os.path.join(results_dir, experiment[0]['save'], f"run{0}"))