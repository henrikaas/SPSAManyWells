"""
Helper functions
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spsa.optimizer import SPSA, SPSAConfig  # only for type checkers; no runtime import
    from manywells.simulator import SSDFSimulator

import pandas as pd
import numpy as np
import os
from datetime import datetime

from manywells.simulator import SimError, WellProperties, BoundaryConditions
from scripts.data_generation.well import Well
import manywells.pvt as pvt
from manywells.inflow import Vogel
from manywells.choke import SimpsonChokeModel
from spsa.constraints import WellSystemConstraints
from scripts.data_generation.file_utils import save_well_config_and_data



def load_well_configs(filepath):
    """
    Load well configuration data from a CSV file.
    """
    dataset = pd.read_csv(filepath)
    return dataset

def configure_wells(filepath) -> list[Well]:
    """
    Loads a well configuration file and configures the wells.

    :return: Dictionary of wells
    """
    config_dataset = load_well_configs(filepath)

    wells = []
    for index, w in config_dataset.iterrows():
        well_properties = WellProperties(
            L = w['wp.L'],
            D = w['wp.D'],
            rho_l = w['wp.rho_l'],
            R_s = w['wp.R_s'],
            cp_g = w['wp.cp_g'],
            cp_l = w['wp.cp_l'],
            f_D = w['wp.f_D'],
            h = w['wp.h'],
        )

        if w['wp.inflow.class_name'] == 'Vogel':
            inflow = Vogel(
                w_l_max = w['wp.inflow.w_l_max'],
                f_g = w['wp.inflow.f_g']
            )
            well_properties.inflow = inflow
        else:
            print('Inflow Model name not "Vogel"')
        
        if w['wp.choke.class_name'] == 'SimpsonChokeModel':
            choke = SimpsonChokeModel(
                K_c = w['wp.choke.K_c'],
                chk_profile = w['wp.choke.chk_profile']
            )
            well_properties.choke = choke
        else:
            print('Choke Model name not SimpsonChokeModel')

        boundary_conditions = BoundaryConditions(
            p_r = w['bc.p_r'],
            p_s = w['bc.p_s'],
            T_r = w['bc.T_r'],
            T_s = w['bc.T_s'],
            u = w['bc.u'],
            w_lg = w['bc.w_lg']
        )

        gas = pvt.GasProperties(
            name = 'gas',
            R_s = w['gas.R_s'],
            cp = w['gas.cp']
        )

        oil = pvt.LiquidProperties(
            name = 'oil',
            rho = w['oil.rho'],
            cp = w['oil.cp']
        )

        water = pvt.WATER

        f_g = w['fraction.gas']
        f_o = w['fraction.oil']
        f_w = w['fraction.water']

        well = Well(wp=well_properties, bc=boundary_conditions,
                    gas=gas, oil=oil, water=water,
                    fractions=(f_g, f_o, f_w),
                    has_gas_lift=w['has_gas_lift']
        )

        if 'x_last' in w:
            guess = w['x_last']
            guess = eval(w['x_last'])  # Convert string representation of list back to list
            well.x_guesses.append(guess)

        wells.append(well)

    return wells

def create_data_point(well: Well, sim: SSDFSimulator, x, sim_type=None):
    """
    Creating new data point to add to well_data.
    Based on generate_well_data.simulate_well

    :param well: Well object
    :param sim: Simulation object
    :param x: Simulation to base the data point on.
    :return: New data point, as df
    """

    # Prepare new data point
    df_x = sim.solution_as_df(x)

    df_x['w_g'] = well.wp.A * df_x['alpha'] * df_x['rho_g'] * df_x['v_g']
    df_x['w_l'] = well.wp.A * (1 - df_x['alpha']) * df_x['rho_l'] * df_x['v_l']

    pbh = float(df_x['p'].iloc[0])
    pwh = float(df_x['p'].iloc[-1])
    twh = float(df_x['T'].iloc[-1])
    w_g = float(df_x['w_g'].iloc[-1])  # Including lift gas
    w_l = float(df_x['w_l'].iloc[-1])
    w_tot = w_g + w_l
    w_lg = well.bc.w_lg

    # Get oil and water mass flow rate
    f_g, f_o, f_w = well.fractions
    wlf = f_w / (f_o + f_w)  # Water to liquid fraction
    w_w = w_l * wlf
    w_o = w_l * (1 - wlf)

    # Volumetric flow rates (at standard reference conditions) in Sm³/s
    rho_g = pvt.gas_density(sim.wp.R_s)
    q_g = w_g / rho_g  # Including lift gas
    q_lg = w_lg / rho_g
    q_l = w_l / well.wp.rho_l
    q_o = w_o / well.oil.rho
    q_w = w_w / well.water.rho
    q_tot = q_g + q_l

    # assert abs(q_l - (q_o + q_w)) < 1e-5, f'Liquids do not sum: q_l = {q_l}, q_o + q_w = {q_o + q_w}'

    # Convert volumetric flow rates from Sm³/s to Sm³/h
    SECONDS_PER_HOUR = 3600
    q_g *= SECONDS_PER_HOUR
    q_lg *= SECONDS_PER_HOUR
    q_l *= SECONDS_PER_HOUR
    q_o *= SECONDS_PER_HOUR
    q_w *= SECONDS_PER_HOUR
    q_tot *= SECONDS_PER_HOUR

    # Choked flow?
    choked = well.wp.choke.is_choked(pwh, well.bc.p_s)

    # Flow regime at top and bottom of well
    regime_wh = str(df_x['flow-regime'].iloc[-1])
    regime_bh = str(df_x['flow-regime'].iloc[0])

    if isinstance(x, (list, np.ndarray, pd.Series)) and np.all(np.asarray(x) == 0.0):
        choked = True
        regime_wh = None
        regime_bh = None

    # Validate data before adding
    valid_rates = w_l >= 0 and w_g >= 0
    valid_fracs = (0 <= f_g <= 1) and (0 <= f_o <= 1) and (0 <= f_w <= 1)
    if not (valid_rates and valid_fracs):
        raise SimError('Flow rates/mass fractions not valid')  # Count failure - discard simulation
        
    # Not applicable when running SPSA optimization
    # # Discard simulation if total mass flow rate is less than 0.1 kg/s
    # if w_l + w_g < 0.1:
    #     # Simulation did not fail, but solution is invalid (too low flow rate)
    #     # n_failed_sim += 1  # Count failure - discard simulation
    #     raise SimError('Total mass flow rate too low')

    # Structure data point in dict
    dp = {
        'CHK': well.bc.u,
        'PBH': pbh,
        'PWH': pwh,
        'PDC': well.bc.p_s,
        'TBH': well.bc.T_r,
        'TWH': twh,
        'WGL': w_lg,
        'WGAS': max(0, w_g - w_lg),  # Excluding lift gas
        'WLIQ': w_l,
        'WOIL': w_o,
        'WWAT': w_w,
        'WTOT': w_tot,  # Total mass flow, including lift gas
        'QGL': q_lg,
        'QGAS': q_g - q_lg,  # Excluding lift gas
        'QLIQ': q_l,
        'QOIL': q_o,
        'QWAT': q_w,
        'QTOT': q_tot,  # Total volumetric flow, including lift gas
        'FGAS': f_g,  # Inflow gas mass fraction (WGAS / (WTOT - WGL))
        'FOIL': f_o,  # Inflow oil mass fraction (WOIL / (WTOT - WGL))
        'FWAT': f_w,  # Inflow water mass fraction (WWAT / (WTOT - WGL))
        'CHOKED': choked,
        'FRBH': regime_bh,  # Flow regime at bottomhole
        'FRWH': regime_wh,  # Flow regime at wellhead
    }
    if sim_type is not None:
        dp['SIM'] = sim_type # Type of desicion (exploring, optimizing)

    # Add new data point to dataset
    new_dp = pd.DataFrame(dp, index=[0])  # Passing index since values are scalar

    return new_dp

def create_sim_results_df():
    """
    Creates df to store simulation results.
    Based on generate_well_data.simulate_well
    """
    cols = ['CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH',
            'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
            'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT',
            'FGAS', 'FOIL', 'FWAT', 'CHOKED', 'FRBH', 'FRWH', 'SIM']
    well_data = pd.DataFrame(columns=cols, dtype=np.float32)
    well_data['CHOKED'] = well_data['CHOKED'].astype(bool)
    well_data['FRBH'] = well_data['FRBH'].astype(str)
    well_data['FRWH'] = well_data['FRWH'].astype(str)

    return well_data

def calculate_state(well_data: pd.DataFrame, sigma: float = 0.0):
    df = well_data

    totals = {
        "WOIL": df["WOIL"].sum(axis=0),
        "WWAT": df["WWAT"].sum(axis=0),
        "WGL": df["WGL"].sum(axis=0),
        "WGAS": df["WGAS"].sum(axis=0),
    }

    measured = totals.copy()
    if sigma > 0:
        for key, value in totals.items():
            std_dev = (sigma / 100.0) * value
            measured[key] = value + np.random.normal(0, std_dev)

    oil = measured["WOIL"]
    water = measured["WWAT"]
    gas_lift = measured["WGL"]
    gas = measured["WGAS"]

    print(f"Measured oil production (with noise): {oil} kg/s")
    print(f"Real oil production: {totals['WOIL']} kg/s")

    print(f"Measured water production (with noise): {water} kg/s")
    print(f"Real water production: {totals['WWAT']} kg/s")

    print(f"Measured Gas-lift production: {gas_lift} kg/s")
    print(f"Real gas-lift production: {totals['WGL']} kg/s")

    print(f"Measured gas production (excl. lift gas): {gas} kg/s")
    print(f"Real gas production: {totals['WGAS']} kg/s")

    return {"oil": oil, "water": water, "gas_lift": gas_lift, "gas": gas}

def save_data(wells: list[Well], well_data: pd.DataFrame, main_path: str, k: int):
    path = f"{main_path}/iteration_{k}"

    for i, well in enumerate(wells):
        save_well_config_and_data(config=well, data=well_data[i], dataset_version=path)

def save_fail_log(path: str, k: int, fails_per_well: dict[list], success: bool):
    """
    Saves a log file detailing the failures encountered during a process.

    Args:
        path (str): The directory path where the log file will be saved.
        k (int): The number of iterations completed before finishing or crashing.
        fails_per_well (dict[list]): A dictionary mapping well identifiers (keys) to lists of failure messages (values).
        success (bool): Indicates whether the process finished successfully (True) or crashed (False).

    The function creates the specified directory if it does not exist, and writes a formatted log file named
    'failures.txt' containing the status, number of iterations, and failure details for each well.
    """
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/failures.txt", "w") as f:
        f.write(f"------- Failure log -------\n")
        status_msg = "Successful finish" if success else "Crashed"
        f.write(f"{status_msg} after {k} iterations.\n")
        f.write(f"---------------------------\n\n")

        for key, fails in fails_per_well.items():
            f.write(f"------ Well L={key:.1f} ------\n")
            f.write(f"Total failures: {len(fails)}\n")
            for line in fails:
                f.write(line)
            f.write(f"---------------------------\n\n")

def append_fail_log(fail_log: list[str], well: Well, k: int):
    fail_msg = f"Failure {len(fail_log) +1} at iteration {k} with u={well.bc.u}, gl={well.bc.w_lg}\n"
    fail_log.append(fail_msg)
    return fail_log

def save_init_log(path: str, description: dict[str, any]):

    os.makedirs(path, exist_ok=True)
    with open(f"{path}/system_description.txt", "w") as f:
        f.write(f"------- System description -------\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"{description['description']}\n")
        f.write(f"----------------------------------\n\n")

        f.write(f"Attempted number of iterations: {description['n_sim']}\n")
        f.write(f"Number of wells: {description['n_wells']}\n")
        f.write(f"Starting decision vector: {description['start']}\n\n")

        constraints: WellSystemConstraints = description['constraints']
        f.write(f"Constraints:\n")
        f.write(f"gl_max: {constraints.gl_max}\n")
        f.write(f"comb_gl_max: {constraints.comb_gl_max}\n")
        f.write(f"wat_max: {constraints.wat_max}\n")
        f.write(f"max_wells: {constraints.max_wells}\n")
        f.write(f"movement max: {constraints.l_max}\n\n")

        hyperparams: SPSAConfig = description['hyperparams']
        f.write(f"SPSA hyperparametres:\n")
        f.write(f"a: {hyperparams.a}\n")
        f.write(f"b: {hyperparams.b}\n")
        f.write(f"c: {hyperparams.c}\n")
        f.write(f"A: {hyperparams.A}\n")
        f.write(f"alpha: {hyperparams.alpha}\n")
        f.write(f"beta: {hyperparams.beta}\n")
        f.write(f"gamma: {hyperparams.gamma}\n")
        f.write(f"sigma: {hyperparams.sigma}\n")
        f.write(f"rho: {hyperparams.rho}\n\n")

        f.write(f"config file: {description['config']}\n")

def create_dirs(experiments: list[dict], n_runs: int):
    """
    Create directories for saving the results of each experiment.
    """
    work_dir = os.environ["USER_WORK"]

    results_dir = os.path.join(work_dir, "results")
    os.makedirs(results_dir, exist_ok=True) # Create results directory if it doesn't exist

    for experiment in experiments:
        save_dir = os.path.join(results_dir, f"{experiment['save']}")
        os.makedirs(save_dir, exist_ok=False)

        for run_idx in range(0, n_runs):
            run_dir = os.path.join(save_dir, f"run{run_idx}")
            os.makedirs(run_dir, exist_ok=True)
    
    return work_dir, results_dir

def choked_flow(well: Well, sample_type:str):
    """
    If choke == 0, we handle it outside of the simulator
    """
    dp = {
    'CHK': well.bc.u,
    'PBH': None,
    'PWH': None,
    'PDC': well.bc.p_s,
    'TBH': well.bc.T_r,
    'TWH': None,
    'WGL': well.bc.w_lg,
    'WGAS': 0.0,  # Excluding lift gas
    'WLIQ': 0.0,
    'WOIL': 0.0,
    'WWAT': 0.0,
    'WTOT': 0.0,  # Total mass flow, including lift gas
    'QGL': 0.0,
    'QGAS': 0.0,  # Excluding lift gas
    'QLIQ': 0.0,
    'QOIL': 0.0,
    'QWAT': 0.0,
    'QTOT': 0.0,  # Total volumetric flow, including lift gas
    'FGAS': 0.0,  # Inflow gas mass fraction (WGAS / (WTOT - WGL))
    'FOIL': 0.0,  # Inflow oil mass fraction (WOIL / (WTOT - WGL))
    'FWAT': 0.0,  # Inflow water mass fraction (WWAT / (WTOT - WGL))
    'CHOKED': True,
    'FRBH': None,  # Flow regime at bottomhole
    'FRWH': None,  # Flow regime at wellhead
    }
    if sample_type is not None:
        dp['SIM'] = sample_type # Type of desicion (exploring, optimizing)

    # Add new data point to dataset
    new_dp = pd.DataFrame(dp, index=[0])  # Passing index since values are scalar
    return new_dp, None
