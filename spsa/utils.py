"""
Helper functions
"""

import pandas as pd
import numpy as np

from manywells.simulator import SimError, WellProperties, BoundaryConditions
from scripts.data_generation.well import Well
import manywells.pvt as pvt
from manywells.inflow import Vogel
from manywells.choke import SimpsonChokeModel

def load_well_configs(filepath):
    """
    Load well configuration data from a CSV file.
    """
    dataset = pd.read_csv(filepath)
    return dataset

def configure_wells(filepath) -> dict[int, Well]:
    """
    Loads a well configuration file and configures the wells.

    :return: Dictionary of wells
    """
    config_dataset = load_well_configs(filepath)

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

def create_data_point(well, sim, x, decision_type=None):
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

    # Validate data before adding
    valid_rates = w_l >= 0 and w_g >= 0
    valid_fracs = (0 <= f_g <= 1) and (0 <= f_o <= 1) and (0 <= f_w <= 1)
    if not (valid_rates and valid_fracs):
        raise SimError('Flow rates/mass fractions not valid')  # Count failure - discard simulation
        
    # Discard simulation if total mass flow rate is less than 0.1 kg/s
    if w_l + w_g < 0.1:
        # Simulation did not fail, but solution is invalid (too low flow rate)
        # n_failed_sim += 1  # Count failure - discard simulation
        raise SimError('Total mass flow rate too low')

    # Structure data point in dict
    dp = {
        'CHK': well.bc.u,
        'PBH': pbh,
        'PWH': pwh,
        'PDC': well.bc.p_s,
        'TBH': well.bc.T_r,
        'TWH': twh,
        'WGL': w_lg,
        'WGAS': w_g - w_lg,  # Excluding lift gas
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
    if decision_type is not None:
        dp['DCNT'] = decision_type # Type of desicion (exploring, optimizing)

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
            'FGAS', 'FOIL', 'FWAT', 'CHOKED', 'FRBH', 'FRWH', 'DCNT']
    well_data = pd.DataFrame(columns=cols, dtype=np.float32)
    well_data['CHOKED'] = well_data['CHOKED'].astype(bool)
    well_data['FRBH'] = well_data['FRBH'].astype(str)
    well_data['FRWH'] = well_data['FRWH'].astype(str)

    return well_data