"""
Script for testing a set of wells for different values of gas lift and choke settings.
"""

import pandas as pd
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from manywells.simulator import SSDFSimulator
from scripts.data_generation.file_utils import save_well_config_and_data
from scripts.data_generation.well import Well

from manywells.simulator import SSDFSimulator, SimError, WellProperties, BoundaryConditions
from manywells.slip import SlipModel
from manywells.inflow import InflowModel, Vogel
from manywells.choke import ChokeModel, SimpsonChokeModel
from scripts.data_generation.well import Well, sample_well
import manywells.pvt as pvt

PERT_LENGTH = 0.1 # Length of perturbation
PERT_DIRECTIONS = [(1,1), (1,-1), (-1,1), (-1,-1)]
ITERATIONS = 5 # Number of iterations

U_MIN = 0
U_MAX = 1
LG_MIN = 0
LG_MAX = 5

def configure_wells(config_dataset) -> dict[int, Well]:
    """
    Configures the wells from config file

    :return: Dictionary of wells
    """
    wells = {}
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

        wells[index] = well

    return wells

def sim_well(simulator: SSDFSimulator):
    for i in range(ITERATIONS):
        try:
            dp = simulator.simulate()
            return dp
        except SimError as e:
            print(f'Failed simulating {i+1} time - Simulation error: {e}. Retrying...')
            continue
    print(f'\n====================\nSimulation failed after maximum retries.\n====================\n')
    return None

def gradient_length(i):
    """a_k: gradient length per iteration."""
    return 0.1 / (i + 50) ** 0.301

# ------- Main script -------
filepath = "data/stresstest/"
filename = "optimized_wells_config.csv"
dataset = pd.read_csv(filepath + filename)
wells = configure_wells(dataset)

simulators = []
for id, well in wells.items():
    simulator = SSDFSimulator(well.wp, well.bc)
    simulators.append(simulator)

fails_per_iteration = [0] * ITERATIONS
fails_per_well = [0] * len(wells)
for i in range(ITERATIONS):
    print(f'--- Iteration {i+1} of {ITERATIONS} ---')
    g = gradient_length(i+1)

    for id, well in wells.items():
        orig_u = well.bc.u
        orig_w_lg = well.bc.w_lg

        success = True
        for direction in PERT_DIRECTIONS:
            u = orig_u + PERT_LENGTH * direction[0]
            well.bc.u = np.clip(u, U_MIN, U_MAX)  # Ensure u is within bounds
            lg = orig_w_lg + PERT_LENGTH * direction[1] * LG_MAX
            well.bc.w_lg = np.clip(lg, LG_MIN, LG_MAX)  # Ensure w_lg is within bounds
            
            print(f"Simulating well {id} with u={well.bc.u:.3f}, w_lg={well.bc.w_lg:.3f}")
            dp = sim_well(simulators[id])

            if dp is not None:
                print("Simulation successful.")
            else:
                success = False
                fails_per_iteration[i] += 1
                fails_per_well[id] += 1

        # Find a new decision variable as starting point for next iteration
        # Finds a new decision vector by random walk
        direction = random.choice(PERT_DIRECTIONS)
        u = orig_u + g * direction[0]
        well.bc.u = np.clip(u, U_MIN, U_MAX)
        lg = orig_w_lg + g * direction[1] * LG_MAX
        well.bc.w_lg = np.clip(lg, LG_MIN, LG_MAX)

        print("====================")
        print(f"Simulation successful for well {id} for all perturbations." if success else f"Simulation failed for well {id} for one or more perturbations.")
        print(f"New starting point for well {id}: u={well.bc.u:.3f}, w_lg={well.bc.w_lg:.3f}\n")
    
print("Summary of simulation failures:")
for i, fails in enumerate(fails_per_iteration):
    print(f"Iteration {i+1}: {fails} failures")
for id, fails in enumerate(fails_per_well):
    print(f"Well {id}: {fails} failures")

print(f"Total failures: {sum(fails_per_well)}")