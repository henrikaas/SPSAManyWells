"""
Different functions for visualizing SPSA results.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.environ["RESULTS_DIR"]
PLOT_DIR = os.environ["PLOT_DIR"]

INIT_INFO: dict = {
    "mixed_prod_choke50%": 
        {"oil": 62.636,
         "water": 19.594,
         "gaslift": 0.0,
         "water_max": 20.0,
         "gaslift_max": 10.0,
         "opt_prod": 68},}

sns.set_style("darkgrid")
plt.rcParams.update({'font.size':16})

# -------------- Helper functions -----------------
def extract_settings(experiment_dir: Path) -> dict:
    """
    Reads system_description.txt file from experiment directory. Returns a dictionary with the settings of the system.
    """
    settings_path = experiment_dir / "system_description.txt"
    with open(settings_path, "r") as f:
        text = f.read()

    wells = re.search(r"Number of wells:\s*(\d+)", text)
    constraints_block = re.search(r"Constraints:\s*(.+?)(?:\n\s*\n|$)", text, re.S)

    data = {
        "n_wells": int(wells.group(1)) if wells else None,
        "constraints": {}
    }

    if constraints_block:
        for line in constraints_block.group(1).strip().splitlines():
            k, v = [x.strip() for x in line.split(":", 1)]
            data["constraints"][k] = float(v)
    return data

# -------------- Main plotting functions -----------------

def plot_spsa_experiment(experiment_name: str, config_file: str,
                         only_optimizing_iterations: bool = False,
                         save: bool = False):
    """
    Plots iteration sequence from one SPSA experiment.
    """
    experiment_dir = Path(f"{DATA_DIR}/{experiment_name}")
    info = INIT_INFO[config_file]
    info.update(extract_settings(experiment_dir))

    runs = [r for r in experiment_dir.iterdir() if r.is_dir()]
    n_runs = len(runs)

    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for run_idx, run in enumerate(runs):
        print(f"Processing Run {run_idx}/{n_runs-1}...")
        # Find all CSVs that match the pattern iteration_X/iteration_X.csv
        candidates = list(run.glob("iteration_*/iteration_*.csv"))

        if candidates:
            # Extract iteration numbers from folder names
            def get_iter(p: Path) -> int:
                return int(p.parent.name.split("_")[1])
            
            latest = max(candidates, key=get_iter) # Get the file with the highest iteration number
            df = pd.read_csv(latest)
        else:
            print(f"No valid data found for Run {run_idx}. Skipping.")
            continue
        
        # Number of wells with check
        n_wells = info["n_wells"]
        if n_wells != len(df.groupby('ID')):
            raise ValueError(f"Number of wells in data ({len(df.groupby('ID'))}) does not match expected ({n_wells})")

        # TODO: Check if we need data cleaning
        # df = keep_last_unique_pairs(df, correct_i = 3*i)
        max_wells = info["constraints"].get("max_wells", n_wells)
        n_sims = int(len(df) / (2 * min(max_wells, n_wells) + n_wells))

        if only_optimizing_iterations:
            df = df[df["SIM"] == "Optimizing"]
        else:
            n_sims *= 3  # Each iteration has 3 simulations (neg pert, pos pert, optim)
        
        well_data = df.groupby('ID')

        oil, gasl, water = [info["oil"]], [info["gaslift"]], [info["water"]]

        # TODO: This does not handle that wells can have different lengths of data (if we dont perturb on every well)
        for i in range(n_sims):
            o = g = w = 0.0
            for well_id in range(n_wells):
                well = well_data.get_group(well_id)
                o += well['WOIL'].iloc[i]
                g += well['WGL'].iloc[i]
                w += well['WWAT'].iloc[i]

            oil.append(o)
            gasl.append(g)
            water.append(w)

        # x_vals = range(1, len(oil) + 1)
        axs[0].plot(oil, label=f'Run {run_idx+1}', color='brown', alpha=0.5) # Oil production
        axs[1].plot(gasl, label=f'Run {run_idx+1}', color='green', alpha=0.5) # Gas lift
        axs[2].plot(water, label=f'Run {run_idx+1}', color='blue', alpha=0.5) # Water production

        axs[0].plot(len(oil)-1, oil[-1], '|', color='brown', markersize=4) # Mark final point
        axs[1].plot(len(gasl)-1, gasl[-1], '|', color='green', markersize=4) # Mark final point
        axs[2].plot(len(water)-1, water[-1], '|', color='blue', markersize=4) # Mark final point

    # axs[0].set_ylim(top=110, bottom=95)
    # axs[1].set_ylim(top=10, bottom=0)
    # axs[2].set_ylim(top=78, bottom=38)
    axs[0].set_title('Oil Production')
    axs[1].set_title('Gas Lift')
    axs[2].set_title('Water Production')
    axs[2].set_xlabel('Simulation Steps')

    axs[1].axhline(y=info["constraints"]["comb_gl_max"], color='k', linestyle='--', linewidth=1.5) # Visualize combined gas lift max
    axs[2].axhline(y=info["constraints"]["wat_max"], color='k', linestyle='--', linewidth=1.5) # Visualize water production max

    for ax in axs:
        # ax.legend()
        ax.grid(True)

    # fig.suptitle(fr"Prodcution under no noise : $\sigma = 0$")
    fig.tight_layout()
    if save:
        plt.savefig(f"{PLOT_DIR}/{experiment_name}_prod.png", dpi=300, bbox_inches="tight")

    plt.show()

    # --------------------------------------
       
        # if only_optimizing_iterations:
        #     df = df[2::3]
        # else:
        #     n_sims *= 3

        # df = df[col_names]

# TODO: Make file for printing data from the runs. As in prosjektoppgave, plot_multiple_runs

if __name__ == "__main__":
    plot_spsa_experiment(experiment_name="experiments rho/mixedprod_rho2_water20", config_file="mixed_prod_choke50%", only_optimizing_iterations=False)