"""
Different functions for visualizing SPSA results.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns

DATA_DIR = os.environ["RESULTS_DIR"]
PLOT_DIR = os.environ["PLOT_DIR"]

INIT_INFO: dict = {
    "mixedprod_choke50": {
        "oil": 62.636,
        "water": 19.594,
        "gaslift": 0.0,
        "opt_prod": 68
    },
    "10randomwells": {
         "oil": 169.7,
         "water": 68,
         "gaslift": 0.0,
        #  "opt_prod": 200 # TODO: dont know yet
    },
    "20randomwells": {
        "oil": 197.5,
        "water": 208.5,
        "gaslift": 0.0,
        # "opt_prod": 120 # TODO: dont know yet
    },
}

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

    n_sims = re.search(r"Attempted number of iterations:\s*(\d+)", text)
    wells = re.search(r"Number of wells:\s*(\d+)", text)
    constraints_block = re.search(r"Constraints:\s*(.+?)(?:\n\s*\n|$)", text, re.S)

    cfg = re.search(r"config file:\s*([^\s#]+)", text, re.IGNORECASE)

    data = {
        "n_sims": int(n_sims.group(1)) if n_sims else None,
        "n_wells": int(wells.group(1)) if wells else None,
        "constraints": {},
        "config_file": cfg.group(1).strip() if cfg else None,
    }

    if constraints_block:
        for line in constraints_block.group(1).strip().splitlines():
            k, v = [x.strip() for x in line.split(":", 1)]
            data["constraints"][k] = float(v)
    return data

def list_iterations(experiment_dir: Path) -> list[int]:
    """List all iteration numbers in an experiment, sorted"""
    iters = set()
    for run in experiment_dir.iterdir():
        if not run.is_dir():
            continue
        for p in run.glob("iteration_*"):
            if p.is_dir():
                m = re.match(r"iteration_(\d+)$", p.name)
                if m:
                    iters.add(int(m.group(1)))
    return sorted(iters)

def extract_production_history(data: pd.DataFrame, n_sims: int, init_production: tuple[float,float,float], only_optimizing: bool):
    """
    Extracts the production history from the data provided.
    """
    if only_optimizing: 
        data = data[data['SIM'].isin(['Unselected Well', 'Optimizing'])]

    well_data = data.groupby('ID')
    n_wells = len(well_data)

    init_oil, init_gasl, init_water = init_production
    oil, gasl, water = [init_oil], [init_gasl], [init_water]

    if only_optimizing:
        well_data = well_data
        for i in range(n_sims):
            o = g = w = 0.0
            for well_idx in range(n_wells):
                well = well_data.get_group(well_idx)
                
                o += well['WOIL'].iloc[i]
                g += well['WGL'].iloc[i]
                w += well['WWAT'].iloc[i]

            oil.append(o); gasl.append(g); water.append(w)

    else:
        iteration = [0] * n_wells
        for _ in range(n_sims):
            o = [0.0] * 3
            g = [0.0] * 3
            w = [0.0] * 3
            for well_idx in range(n_wells):
                well = well_data.get_group(well_idx)
                i = iteration[well_idx] # Find the position where we left of in the data

                if well["SIM"].iloc[i] == "Unselected Well": # The same simulation is used for all three steps
                    # Oil
                    o[0] += well['WOIL'].iloc[i]
                    o[1] += well['WOIL'].iloc[i]
                    o[2] += well['WOIL'].iloc[i]
                    # Gas
                    g[0] += well['WGL'].iloc[i]
                    g[1] += well['WGL'].iloc[i]
                    g[2] += well['WGL'].iloc[i]
                    # Water
                    w[0] += well['WWAT'].iloc[i]
                    w[1] += well['WWAT'].iloc[i]
                    w[2] += well['WWAT'].iloc[i]

                    iteration[well_idx] = i + 1
                else:
                    # Oil
                    o[0] += well['WOIL'].iloc[i]
                    o[1] += well['WOIL'].iloc[i+1]
                    o[2] += well['WOIL'].iloc[i+2]
                    # Gas
                    g[0] += well['WGL'].iloc[i]
                    g[1] += well['WGL'].iloc[i+1]
                    g[2] += well['WGL'].iloc[i+2]
                    # Water
                    w[0] += well['WWAT'].iloc[i]
                    w[1] += well['WWAT'].iloc[i+1]
                    w[2] += well['WWAT'].iloc[i+2]

                    iteration[well_idx] = i + 3

            oil += o; gasl += g; water += w

    return oil, gasl, water

def extract_decision_vector(data: pd.DataFrame, only_optimizing: bool = False):
    """
    Extracts the decision vector for each iteration, using the data provided.
    Important: Needs to be a /iteration_*.csv file, not config file.
    """
    if only_optimizing: 
        data = data[data['SIM'].isin(['Unselected Well', 'Optimizing'])]

    well_data = data.groupby('ID')
    n_wells = len(well_data)

    u_vals = []
    gl_vals = []

    for well_idx in range(n_wells):
        well = well_data.get_group(well_idx)
        u_vals.append(well["CHK"].values)
        gl_vals.append(well["WGL"].values)

    return u_vals, gl_vals

# -------------- Main plotting functions -----------------

def plot_spsa_experiment(experiment_name: str,
                         only_optimizing_iterations: bool = False,
                         save: bool = False):
    """
    Plots iteration sequence from one SPSA experiment.
    """
    experiment_dir = Path(f"{DATA_DIR}/{experiment_name}")
    info = extract_settings(experiment_dir)
    config_file = info["config_file"]
    info.update(INIT_INFO[config_file])

    runs = [r for r in experiment_dir.iterdir() if r.is_dir()]
    n_runs = len(runs)

    n_wells = info["n_wells"]
    max_wells = info["constraints"].get("max_wells", n_wells)

    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    for run_idx, run in enumerate(runs):
        print(f"Processing Run {run_idx}/{n_runs-1}...")
        # Find all CSVs that match the pattern iteration_X/iteration_X.csv
        candidates = [
            p for p in run.glob("iteration_*/iteration_*.csv")
            if p.stem == p.parent.name  # ensures iteration_50/iteration_50.csv
        ]

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
        if n_wells != len(df.groupby('ID')):
            raise ValueError(f"Number of wells in data ({len(df.groupby('ID'))}) does not match expected ({n_wells})")

        # TODO: Check if we need data cleaning
        # df = keep_last_unique_pairs(df, correct_i = 3*i)
        n_sims = int(len(df) / (2 * min(max_wells, n_wells) + n_wells))
        
        oil, gasl, water = extract_production_history(data=df, 
                                                      n_sims=n_sims, 
                                                      init_production=(info["oil"], info["gaslift"], info["water"]),
                                                      only_optimizing=only_optimizing_iterations)

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
    fig.suptitle(experiment_name)
    if save:
        plt.savefig(f"{PLOT_DIR}/{experiment_name}_prod.png", dpi=300, bbox_inches="tight")

    plt.show()

def plot_decision_vector(experiment_name: str, save: bool = False, iteration: int | None = None):
    """
    Plots the decision vector in the last iteration in a 2D-graph
    Choke (bc.u) on x-axis vs Gas Lift (bc.w_lg) on y-axis.
    Each well (identified by 'wp.L') is plotted with a consistent color.
    """
    experiment_dir = Path(f"{DATA_DIR}/{experiment_name}")
    info = extract_settings(experiment_dir)

    runs = [r for r in experiment_dir.iterdir() if r.is_dir()]
    n_runs = len(runs)

    # Decide which iteration to show
    if iteration is None:
        iteration = info["n_sims"]

    fig, ax = plt.subplots()
    well_ids = set()
    dfs = []

    for run_idx, run in enumerate(runs):
        print(f"Processing Run {run_idx}/{n_runs-1} (iteration {iteration})...")
        cfg_path = Path(f"{run}/iteration_{iteration}/iteration_{iteration}_config.csv")
        if not cfg_path.exists():
            print(f"No valid data found for Run {run_idx} (file missing). Skipping.")
            continue

        df = pd.read_csv(cfg_path)
        well_ids.update(df['wp.L'].unique())
        dfs.append(df)


    # Assign a unique color to each well using wp.L
    well_ids = sorted(well_ids)  # Optional: consistent order
    colormap = cm.get_cmap('tab20', len(well_ids))
    color_mapping = {well_id: colormap(i % 10) for i, well_id in enumerate(well_ids)}

    for df in dfs:
        for _, row in df.iterrows():
            well_id = row['wp.L']
            u = row['bc.u']
            w_lg = row['bc.w_lg']
            color = color_mapping[well_id]

            ax.scatter(u, w_lg, color=color, label=f"Well {well_ids.index(well_id)}" if run_idx == 0 else "", s=30)

    ax.set_ylim(-0.1, info["constraints"].get("gl_max", None))
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel("Choke")
    ax.set_ylabel("Gas Lift")
    # ax.set_title("Decision Iterate per Well Across Runs")

    legend_patches = [
        mpatches.Patch(color=color_mapping[well_id], label=f"Well {i}")
        for i, well_id in enumerate(well_ids)
    ]

    ax.legend(handles=legend_patches, fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.title(experiment_name)
    plt.tight_layout()
    if save:
        plt.savefig(f"{PLOT_DIR}/{experiment_name}_decvector.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_decision_vector_series(experiment_name: str, save_each: bool = False, start: int | None = None, stop: int | None = None):
    """
    Finds iterations in experiment_name, sorts ascending, and calls plot_decision_vector for each.
    Makes it possible to track the decision vector over time.
    Optional start/stop (inclusive bounds) to restrict the range.
    """
    experiment_dir = Path(f"{DATA_DIR}/{experiment_name}")
    all_iters = list_iterations(experiment_dir)
    if not all_iters:
        raise FileNotFoundError("No iteration_* folders found.")

    # restrict range if asked
    iters = [it for it in all_iters if (start is None or it >= start) and (stop is None or it <= stop)]

    for it in iters:
        plot_decision_vector(experiment_name, save=save_each, iteration=it)

def print_production_sequence(experiment_name: str):
    """
    Prints the production sequence from one SPSA experiment.
    """
    experiment_dir = Path(f"{DATA_DIR}/{experiment_name}")
    info = extract_settings(experiment_dir)
    config_file = info["config_file"]
    info.update(INIT_INFO[config_file])

    runs = [r for r in experiment_dir.iterdir() if r.is_dir()]
    n_runs = len(runs)

    n_wells = info["n_wells"]
    max_wells = info["constraints"].get("max_wells", n_wells)

    for run_idx, run in enumerate(runs):
        print(f"Processing Run {run_idx}/{n_runs-1}...")
        # Find all CSVs that match the pattern iteration_X/iteration_X.csv
        candidates = [
            p for p in run.glob("iteration_*/iteration_*.csv")
            if p.stem == p.parent.name  # ensures iteration_50/iteration_50.csv
        ]

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
        if n_wells != len(df.groupby('ID')):
            raise ValueError(f"Number of wells in data ({len(df.groupby('ID'))}) does not match expected ({n_wells})")
        
        n_sims = int(len(df) / (2 * min(max_wells, n_wells) + n_wells))
        
        oil, gasl, water = extract_production_history(data=df, 
                                                      n_sims=n_sims, 
                                                      init_production=(info["oil"], info["gaslift"], info["water"]),
                                                      only_optimizing=False)
        u, gl = extract_decision_vector(data=df, only_optimizing=False)
        
        print(f"Run {run_idx+1} Production Sequence:")
        for i in range(n_sims):
            print(f"====== Iteration {i+1} ======")
            print("--- Positive perturbation ---")
            print(f"Oil: {oil[3*i+1]:.3f}, Gas Lift: {gasl[3*i+1]:.3f}, Water: {water[3*i+1]:.3f}")
            for well_idx in range(n_wells):
                print(f"Well {well_idx+1}: Choke: {u[well_idx][3*i]:.3f}, Gas Lift: {gl[well_idx][3*i]:.3f}")

            print("--- Negative perturbation ---")
            print(f"Oil: {oil[3*i+2]:.3f}, Gas Lift: {gasl[3*i+2]:.3f}, Water: {water[3*i+2]:.3f}")
            for well_idx in range(n_wells):
                print(f"Well {well_idx+1}: Choke: {u[well_idx][3*i+1]:.3f}, Gas Lift: {gl[well_idx][3*i+1]:.3f}")

            print("--- Resulting state ---")
            print(f"Oil: {oil[3*i+3]:.3f}, Gas Lift: {gasl[3*i+3]:.3f}, Water: {water[3*i+3]:.3f}")
            for well_idx in range(n_wells):
                print(f"Well {well_idx+1}: Choke: {u[well_idx][3*i+2]:.3f}, Gas Lift: {gl[well_idx][3*i+2]:.3f}")
            print(f"========================\n")


if __name__ == "__main__":
    plot_spsa_experiment(experiment_name="experiments rho v2/rho8_water20", only_optimizing_iterations=True)
    # plot_decision_vector(experiment_name="experiments rho/mixedprod_rho2_water20")
    # plot_decision_vector_series(experiment_name="experiments gl constraints/mixedprod_strict_comb_gl")
    # print_production_sequence(experiment_name="experiments rho v2/relaxed")

    # ======= Run this if you want to see a set of experiments within a main folder =======
    main_exp = "experiments rho v2" # Change this as needed
    # main_exp = "experiments gl constraints"
    # main_exp = "experiments maxwells"

    main_path = Path(f"{os.environ['RESULTS_DIR']}/{main_exp}")
    experiments = [e for e in main_path.iterdir() if e.is_dir()]

    for exp in experiments:
        plot_spsa_experiment(experiment_name=f"{main_exp}/{exp.name}", only_optimizing_iterations=True)
        plot_decision_vector(experiment_name=f"{main_exp}/{exp.name}")