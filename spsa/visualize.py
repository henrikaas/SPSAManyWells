"""
Different functions for visualizing SPSA results.
"""
import os
import re
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib import colors
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

from gradient import SPSAGradient
from constraints import WellSystemConstraints

DATA_DIR = os.environ["RESULTS_DIR"]
PLOT_DIR = os.environ["PLOT_DIR"]

INIT_INFO: dict = {
    "mixedprod_choke50": {
        "oil": 62.636,
        "water": 19.594,
        "gaslift": 0.0,
        "opt_prod": 68,
        "starting_vector": {"337.29": [0.5, 0], "416.00": [0.5, 0], "371.33": [0.5, 0], "381.84": [0.5, 0], "338.04": [0.5, 0]},
    },
    "10randomwells": {
         "oil": 169.7,
         "water": 68,
         "gaslift": 0.0,
         "starting_vector": [[0.5, 0] for _ in range(10)],
        #  "opt_prod": 200 # TODO: dont know yet
    },
    "20randomwells": {
        "oil": 197.5,
        "water": 208.5,
        "gaslift": 0.0,
        # "opt_prod": 120 # TODO: dont know yet
    },
    "mixedprod_optchoke20_1": {
        "oil": 62.636,
        "water": 19.594,
        "gaslift": 0.0,
        "opt_prod": 68,
    },
    "mixedprod_optchoke20_2": {
        "oil": 62.636,
        "water": 19.594,
        "gaslift": 0.0,
        "opt_prod": 68,
    },
    "mixedprod_optchoke15": {
        "oil": 62.636,
        "water": 19.594,
        "gaslift": 0.0,
        "opt_prod": 68,
        # TODO: "starting_vector": {"337.29": [0.15, 0], "416.00": [0.15, 0], "371.33": [0.15, 0], "381.84": [0.15, 0], "338.04": [0.15, 0]},
    },
    "mixedprod_optchoke10": {
        "oil": 62.636,
        "water": 19.594,
        "gaslift": 0.0,
        "opt_prod": 68,
    },
}


# Seaborn style
sns.set_style("darkgrid")
# sns.set_context("talk")

# # Update Matplotlib rcParams for finer control
plt.rcParams.update({
    "font.size": 16,              # Global font size
    'font.family': 'serif',       # Use a serif font
    'axes.linewidth': 1,          # Thinner axes
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.top': False,
    'ytick.right': False,
    # 'axes.grid': False,         # Remove grid lines (for contour plots)
})

CUSTOM_RC = {
    "font.size": 16,
    "font.family": "serif",
}

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
    params_block = re.search(r"SPSA hyperparametres:\s*(.+?)(?:\n\s*\n|$)", text, re.S)

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
    if params_block:
        for line in params_block.group(1).strip().splitlines():
            k, v = [x.strip() for x in line.split(":", 1)]
            try:
                data[k] = float(v)
            except ValueError:
                data[k] = v
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
        u = []
        gl = []
        for _, row in well.iterrows():
            if row['SIM'] == "Unselected Well":
                u += [row['CHK']]*3
                gl += [row['WGL']]*3
            else:
                u.append(row['CHK'])
                gl.append(row['WGL'])
        u_vals.append(u)
        gl_vals.append(gl)

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
    fig, axs = plt.subplots(3, 1, figsize=(13.33, 7.5), sharex=True, constrained_layout=True)

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
            n_sims = get_iter(latest)
        else:
            print(f"No valid data found for Run {run_idx}. Skipping.")
            continue
        
        # Number of wells with check
        if n_wells != len(df.groupby('ID')):
            raise ValueError(f"Number of wells in data ({len(df.groupby('ID'))}) does not match expected ({n_wells})")

        # TODO: Check if we need data cleaning
        # df = keep_last_unique_pairs(df, correct_i = 3*i)
        # n_sims = int(len(df) / (2 * min(max_wells, n_wells) + n_wells))
        
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

    # axs[0].set_ylim(top=70)
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

def plot_average_production(experiments: list[Path],
                            production_types: list[str] | None = ['oil', 'gas-lift', 'water'],
                            iterations: int = 50,
                         only_optimizing_iterations: bool = False,
                         save: bool = False):
    """
    Compares a set of given experiments by plotting the average production over all runs.
    """
    fig, axs = plt.subplots(len(production_types), 1, figsize=(13.33, 7.5), sharex=True, constrained_layout=True)
    experiments = sorted(experiments, key=lambda e: float(re.search(r'rho(\d+(?:\.\d+)?)', e.name).group(1)))

    for experiment_dir in experiments:
        info = extract_settings(experiment_dir)
        config_file = info["config_file"]
        info.update(INIT_INFO[config_file])

        runs = [r for r in experiment_dir.iterdir() if r.is_dir()]
        n_runs = len(runs)

        n_wells = info["n_wells"]

        oil_prods, gasl_prods, water_prods = [], [], []
        for run_idx, run in enumerate(runs):
            path = Path(f"{run}/iteration_{iterations}/iteration_{iterations}.csv")
            if not path.exists():
                print(f"No valid data found for Run {run_idx} (file missing). Skipping.")
                continue

            df = pd.read_csv(path)
            n_sims = iterations
            
            # Number of wells with check
            if n_wells != len(df.groupby('ID')):
                raise ValueError(f"Number of wells in data ({len(df.groupby('ID'))}) does not match expected ({n_wells})")

            oil, gasl, water = extract_production_history(data=df, 
                                                        n_sims=n_sims, 
                                                        init_production=(info["oil"], info["gaslift"], info["water"]),
                                                        only_optimizing=only_optimizing_iterations)
            oil_prods.append(oil)
            gasl_prods.append(gasl)
            water_prods.append(water)
        
        if "rho" in experiment_dir.name and "water" in experiment_dir.name:
            label=experiment_dir.name.split("_")[0].replace("rho", "")
            label = f"ρ = {label}"

        for i, prod_type in enumerate(production_types):
            if prod_type == 'oil':
                prod_array = np.array(oil_prods)
                avg_prod = np.mean(prod_array, axis=0)
                axs[i].plot(avg_prod, label=label if i == 0 else "", alpha=0.8)
                print(f"Final average oil production for {experiment_dir.name}: {avg_prod[-1]:.2f}")
            elif prod_type == 'gas-lift':
                prod_array = np.array(gasl_prods)
                avg_prod = np.mean(prod_array, axis=0)
                axs[i].plot(avg_prod, label=label if i == 0 else "", alpha=0.8)
                print(f"Final average gas-lift production for {experiment_dir.name}: {avg_prod[-1]:.2f}")
            elif prod_type == 'water':
                prod_array = np.array(water_prods)
                avg_prod = np.mean(prod_array, axis=0)
                axs[i].plot(avg_prod, label=label if i == 0 else "", alpha=0.8)
                print(f"Final average water production for {experiment_dir.name}: {avg_prod[-1]:.2f}")
    if "oil" in production_types:
        axs[production_types.index("oil")].set_ylim(bottom=62, top=70) # These needs to be set manually, water = 20
        # axs[production_types.index("oil")].set_ylim(bottom=40, top=65) # These needs to be set manually, water = 15

        axs[production_types.index("oil")].plot(0, info["oil"],
            marker='o',
            markersize=3,
            color="k",
            alpha=0.6,
            label="_nolegend_")
    if "gas-lift" in production_types:
        axs[production_types.index("gas-lift")].set_ylim(bottom=0, top=11) # These needs to be set manually

        axs[production_types.index("gas-lift")].plot(0, info["gaslift"],
            marker='o',
            markersize=3,
            color="k",
            alpha=0.6,
            label="_nolegend_")
    if "water" in production_types:
        axs[production_types.index("water")].set_ylim(bottom=18.5, top=21.5) # These needs to be set manually, water = 20
        # axs[production_types.index("water")].set_ylim(bottom=5, top=21) # These needs to be set manually, water = 15

        axs[production_types.index("water")].plot(0, info["water"],
            marker='o',
            markersize=3,
            color="k",
            alpha=0.6,
            label="_nolegend_")

    for i, prod_type in enumerate(production_types):

        ymin, ymax = axs[i].get_ylim()
        if prod_type == 'gas-lift':
            bound = info["constraints"]["comb_gl_max"]
            axs[i].axhline(y=bound, color='k', linestyle='-', linewidth=1.25, label="Gas-Lift Boundary") # Visualize combined gas lift max
            axs[i].axhspan(bound, ymax, facecolor="rosybrown", alpha=0.3, zorder=0)
            axs[i].legend(loc="lower right")
        elif prod_type == 'water':
            bound = info["constraints"]["wat_max"]
            axs[i].axhline(y=bound, color='k', linestyle='-', linewidth=1.25, label="Water Production Boundary") # Visualize water production max
            axs[i].axhspan(bound, ymax, facecolor="rosybrown", alpha=0.3, zorder=0)
            axs[i].legend(loc="lower right")
        axs[i].set_title(f'{prod_type.capitalize()} Production')
    axs[len(production_types)-1].set_xlabel('Iterations')
    axs[len(production_types)-1].set_xlim(left=-1, right=iterations + 1)
            

    # =========================

    for ax in axs:
        # ax.legend()
        ax.grid(True)

    # fig.suptitle(fr"Prodcution under no noise : $\sigma = 0$")
    # fig.suptitle(experiment_name)
    axs[0].legend(loc="lower right")
    if save:
        save_dir = f"{PLOT_DIR}/{experiments[0].parent.name}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/avgprod.png", dpi=300, bbox_inches="tight")

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
    base_cmap = plt.get_cmap('tab20')
    colormap = base_cmap(np.linspace(0, 1, len(well_ids)))
    color_mapping = {well_id: colormap[i] for i, well_id in enumerate(well_ids)}

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
            if p.stem == p.parent.name 
        ]

        if candidates:
            # Extract iteration numbers from folder names
            def get_iter(p: Path) -> int:
                return int(p.parent.name.split("_")[1])
            
            latest = max(candidates, key=get_iter) # Get the file with the highest iteration number
            df = pd.read_csv(latest)
            n_sims = get_iter(latest)
        else:
            print(f"No valid data found for Run {run_idx}. Skipping.")
            continue
        
        # Number of wells with check
        if n_wells != len(df.groupby('ID')):
            raise ValueError(f"Number of wells in data ({len(df.groupby('ID'))}) does not match expected ({n_wells})")
        
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

def plot_decision_vector_history(experiment_name: str,
                                only_optimizing_iterations: bool = True,
                                wells_to_plot: list[int] | None = None,
                                iteration: int | None = None,
                                runs: list[int] | None = None,
                                type: str = 'scatter',
                                save: bool = False):
    """
    Plots the decision vector history as a line plot over all iterations for one experiment.
    Choke (bc.u) on x-axis vs Gas Lift (bc.w_lg) on y-axis.
    Each well is plotted with a consistent color.
    If wells_to_plot is provided, only those wells are plotted (by index).

    Args:
        experiment_name: Name of the experiment folder.
        only_optimizing_iterations: If True, only include iterations labeled as 'Optimizing or 'Unselected Well'.
        wells_to_plot: List of well indices to plot. If None, plot all wells.
        iteration: Specific iteration to plot. If None, use the last iteration.
        runs: Which runs to process. If None, process all runs.
        type: Type of plot ('scatter' or 'line').
        save: If True, save the plot to PLOT_DIR.
    """
    experiment_dir = Path(f"{DATA_DIR}/{experiment_name}")
    info = extract_settings(experiment_dir) # Extract settings from experiment description
    config_file = info["config_file"]
    info.update(INIT_INFO[config_file])
    
    if runs is None:
        run_dirs = [r for r in experiment_dir.iterdir() if r.is_dir()]
    else:
        run_dirs = [
            r
            for i in runs
            for r in experiment_dir.glob(f"run{i}")
            if r.is_dir()
        ]

    n_runs = len(run_dirs)
    n_wells = len(wells_to_plot) if wells_to_plot is not None else info["n_wells"]

    # Decide which iteration to show
    if iteration is None:
        iteration = info["n_sims"]

    fig = plt.figure(figsize=(7.5, 7.5), constrained_layout=True)

    colors_assigned = False
    well_ids = set()
    well_cmaps = ["Reds", "Blues", "Greens", "Oranges", "Purples", "Greys", "YlOrRd", "YlOrBr", "YlGn", "PuRd"]
    for run_idx, run in enumerate(run_dirs):
        print(f"Processing Run {run_idx}/{n_runs-1} (iteration {iteration})...")
        print(f"Run path: {run}")
        path = Path(f"{run}/iteration_{iteration}/iteration_{iteration}.csv")
        if not path.exists():
            print(f"No valid data found for Run {run_idx} (file missing). Skipping.")
            continue

        df = pd.read_csv(path)

        if not colors_assigned:
            well_ids.update(df['TBH'].unique())
            # Assign a unique color to each well using TBH
            well_ids = sorted(well_ids)  # Consistent order

            # Change id of wells_to_plot from index to TBH
            if wells_to_plot is not None:
                wells_to_plot = [well_ids[i] for i in wells_to_plot]
            else:
                wells_to_plot = list(well_ids)

            color_mapping = {well_id: well_cmaps[i % len(well_cmaps)] for i, well_id in enumerate(wells_to_plot)}
            colors_assigned = True

        if only_optimizing_iterations: 
            df = df[df['SIM'].isin(['Unselected Well', 'Optimizing'])]
            
        well_data = df.groupby('TBH')
        
        for well_idx in wells_to_plot if wells_to_plot is not None else range(n_wells):
            well = well_data.get_group(well_idx)
            cmap = plt.colormaps[color_mapping[well["TBH"].iloc[0]]]
            starting_vector = info["starting_vector"][f"{well_idx:.2f}"]
            u = [starting_vector[0]] + well["CHK"].tolist()
            gl = [starting_vector[1]] + well["WGL"].tolist()

            n_points = len(u)
            for i in range(n_points):
                color = cmap(0.2 + 0.85 * i / (n_points - 1))  # 0.2→1 avoids very light tones
                if type == 'scatter':
                    plt.scatter(u[i], gl[i], color=color, s=20, label=f"Well {well_idx}" if run_idx == 0 and i == (n_points - 1) // 2 else "")
                elif type == 'line':
                    plt.plot(u[i:i+2], gl[i:i+2],
                        color=color,
                        linewidth=3.5,
                        label=f"Well {well_idx}" if run_idx == 0 and i == (n_points - 1) // 2 else "")


    u_min, u_max = 0.0, 1.0
    gl_min, gl_max = 0.0, info["constraints"].get("gl_max", None)
    if gl_max is None:
        raise ValueError("gl_max constraint not found in experiment settings.")

    plt.xlim(u_min-0.05, u_max+0.05)
    plt.ylim(gl_min-0.05, gl_max+0.05)

    ax = plt.gca()
    ax.add_patch(
        Rectangle(
            (u_min, gl_min),
            u_max - u_min,
            gl_max - gl_min,
            facecolor="none",       # change to e.g. 'tab:green' with alpha if you want fill
            edgecolor="black",      # outline color
            linewidth=2.5,          # thicker outline
            linestyle="--",         # dashed outline; change to '-' for solid
            zorder=5,                # put outline above lines; lower if you want it behind
            alpha=0.3,                # opacity of the outline
            label="Boundaries of the Feasible Region"  # label for legend
        )
    )

    plt.xlabel('Choke')
    plt.ylabel('Gas lift')
    # plt.title('History of the Decision Vector')
    plt.legend()

    # fig.suptitle(experiment_name)
    if save:
        plt.savefig(f"{PLOT_DIR}/{experiment_name}_prod.png", dpi=300, bbox_inches="tight")

    plt.show()

def plot_step_size(experiment_name: str, n_runs: int | None = 10, iteration: int | None = 50, save: bool = False):
    """
    Plots the step size of the SPSA algorithm over iterations.
    Enhanced version:
      - Smooths percentile curves
      - Removes vertical lines between percentiles
      - Adds black y-axis
      - Adds one colored run using YlOrRd colormap
    """
    experiment_dir = Path(f"{DATA_DIR}/{experiment_name}")
    info = extract_settings(experiment_dir)
    config_file = info["config_file"]
    info.update(INIT_INFO[config_file])

    runs = [r for r in experiment_dir.iterdir() if r.is_dir()]
    n_runs = min(n_runs, len(runs)) if n_runs is not None else len(runs)
    runs = runs[:n_runs]

    if iteration is None:
        iteration = info["n_sims"]

    step_sizes_for_all_runs = []

    # --- Gather data from runs ---
    for run_idx, run in enumerate(runs):
        print(f"Processing Run {run_idx}/{n_runs-1} (iteration {iteration})...")
        print(run)
        path = Path(f"{run}/iteration_{iteration}/iteration_{iteration}.csv")
        if not path.exists():
            print(f"No valid data found for Run {run_idx} (file missing). Skipping.")
            continue

        df = pd.read_csv(path)
        df = df[df['SIM'] == 'Optimizing']
        wells = df.groupby('ID')
        n_wells = len(wells)

        step_sizes = []
        for well_idx in range(n_wells):
            well = wells.get_group(well_idx)
            starting_vector = info["starting_vector"][f"{well['TBH'].iloc[0]:.2f}"]
            choke_vals = np.array([starting_vector[0]] + well["CHK"].tolist())
            step_sizes.append(np.abs(np.diff(choke_vals)))

        step_sizes = np.array(step_sizes).max(axis=0)
        step_sizes_for_all_runs.append(step_sizes)

    step_sizes_for_all_runs = np.array(step_sizes_for_all_runs)
    mean_steps = np.mean(step_sizes_for_all_runs, axis=0)
    percentiles_25 = np.percentile(step_sizes_for_all_runs, 25, axis=0)
    percentiles_75 = np.percentile(step_sizes_for_all_runs, 75, axis=0)

    # --- Smooth the percentile boundaries for better visual appeal ---
    smooth_25 = gaussian_filter1d(percentiles_25, sigma=1.0)
    smooth_75 = gaussian_filter1d(percentiles_75, sigma=1.0)
    smooth_mean = gaussian_filter1d(mean_steps, sigma=1.0)

    # --- Start plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = np.arange(1, len(mean_steps)+1)

    # Smooth shaded area (no vertical line segments)
    ax.fill_between(iterations, smooth_25, smooth_75, color='grey', alpha=0.4, label='25–75 Percentile Range')

    # Average step size (smoothed)
    ax.plot(iterations, smooth_mean, color='grey', linewidth=2.5, label='Average Step Size')

    # --- Plot one example run with YlOrRd gradient ---
    example_run = step_sizes_for_all_runs[0]  # or choose another index
    cmap = plt.colormaps['YlOrRd']
    n_points = len(example_run)

    for i in range(n_points - 1):
        color = cmap(0.2 + 0.85 * i / (n_points - 1))
        ax.plot(iterations[i:i+2], example_run[i:i+2], color=color, linewidth=2, label=f"Example Run Step Size" if i == n_points//2 else "")

    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(0, color='black', linewidth=2)
    ax.relim()        # Recalculate limits based on all artists
    ax.autoscale()
    ax.set_xlim(0.95, len(iterations))
    ymax = max(np.max(smooth_75), np.max(example_run)) * 1.1
    ax.set_ylim(-0.0005, ymax)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step Size")
    # ax.set_title(f"Step Sizes for experiments {experiment_name}")

    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"{PLOT_DIR}/{experiment_name}_stepsize.png", dpi=300, bbox_inches="tight")

    plt.show()

def plot_multiple_function_landscapes(experiment_name: str, wells: list[int] | None = None, sigma: float = 0.0, normalize: str = "None", objective: list[str] = ["WOIL"], save: bool = False):
    """
    Plots function landscapes for multiple wells in an experiment.
    """
    def plot_contour_function_landscape(
        well: pd.DataFrame,
        sigma=0.0,
        normalize: str = "None",
        maxmax: float | None = None,
        minmin: float | None = None,
        save: bool = False,
    ):
        """
        Visualizes the function landscape (WGL vs CHK vs WOIL) using contourf.
        Supports local or global normalization and optional Gaussian smoothing.
        """
        if normalize == "global" and (maxmax is None or minmin is None):
            raise ValueError("For global normalization, maxmax and minmin must be provided.")

        # 1. Sort by CHK and WGL
        df_sorted = well.sort_values(by=["WGL", "CHK"])
        df_sorted["OBJ"] = df_sorted[objective].sum(axis=1)

        # 2. Get unique x and y coordinates
        x_vals = np.sort(df_sorted["CHK"].unique())
        y_vals = np.sort(df_sorted["WGL"].unique())

        # 3. Pivot to form 2D grid
        grid = df_sorted.pivot_table(index="WGL", columns="CHK", values="OBJ")
        grid = grid.interpolate(method="linear", axis=0).interpolate(method="linear", axis=1)
        data = grid.values

        print("Grid shape:", data.shape)

        # --- Step 2: Normalize ---
        if normalize == "local":
            data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
            vmin, vmax = None, None
        elif normalize == "global":
            data = (data - minmin) / (maxmax - minmin)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = None, None

        # --- Step 3: Smooth ---
        data = gaussian_filter(data, sigma=sigma)

        # --- Step 4: Contour plot ---
        with sns.axes_style(None):
            plt.figure(figsize=(6,6))
            low_cut = np.nanpercentile(data, 2)

            contour = plt.contourf(
                x_vals,
                y_vals,
                data,
                levels=60,                # number of contour levels
                cmap="viridis",
                vmin=vmin, # or low_cut
                vmax=vmax,
            )

            # Add grey grid lines (optional, coordinate-aware)
            # for x in x_vals:
            #     plt.axvline(x, color="grey", linewidth=0.5, alpha=0.4)
            # for y in y_vals:
            #     plt.axhline(y, color="grey", linewidth=0.5, alpha=0.4)

            # plt.colorbar(contour, label="Normalized Value")
            # plt.title("Smoothed Contour Map (Viridis)")
            plt.xlabel("Choke")
            plt.ylabel("Gas lift")
            plt.tight_layout()
            if save:
                save_dir = f"{PLOT_DIR}/{experiment_name}"
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(f"{save_dir}/well{well['ID'].iloc[0]}_{'_'.join(objective)}.png", dpi=300, bbox_inches="tight")
            plt.show()
    
    experiment_dir = Path(f"{DATA_DIR}/{experiment_name}")
    runs = [r for r in experiment_dir.iterdir() if r.is_dir()]
    n_runs = len(runs)

    for run_idx, run in enumerate(runs):
        print(f"Processing Run {run_idx}/{n_runs-1}...")
        path = Path(f"{run}/iteration_0/iteration_0.csv")
        if not path.exists():
            print(f"No valid data found for Run {run_idx} (file missing). Skipping.")
            continue

        df = pd.read_csv(path)
        maxmax = df['WOIL'].max()
        minmin = df['WOIL'].min()
        wells_grouped = df.groupby('ID')

        if wells is None:
            wells = list(wells_grouped.groups.keys())

        for well_idx in wells:
            well = wells_grouped.get_group(well_idx)
            print(f"Plotting landscape for Well {well_idx} in Run {run_idx}...")

            with plt.rc_context(mpl.rcParamsDefault):
                with plt.rc_context(CUSTOM_RC):
                    plot_contour_function_landscape(well, sigma=sigma, normalize=normalize, maxmax=maxmax, minmin=minmin, save=save)

def plot_mean_function_landscape(
    experiment_name: str,
    wells: list[int] | None = None,
    sigma: float = 0.0,
    normalize: str = "None",
    objective: list[str] = ["WOIL"],
    save: bool = False,
    ):
    """
    Computes and plots the mean normalized function landscape across all wells.
    """
    def compute_normalized_grid(well, sigma=0.0, normalize="None", maxmax=None, minmin=None, objective=["WOIL"]):
        """Helper: compute the normalized smoothed grid for a single well."""
        df_sorted = well.sort_values(by=["WGL", "CHK"])
        df_sorted["OBJ"] = df_sorted[objective].sum(axis=1)

        # pivot to 2D grid
        grid = df_sorted.pivot_table(index="WGL", columns="CHK", values="OBJ")
        grid = grid.interpolate(method="linear", axis=0).interpolate(method="linear", axis=1)
        data = grid.values

        try:
            # normalize
            if normalize == "local":
                data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
            elif normalize == "global":
                if maxmax is None or minmin is None:
                    raise ValueError("Global normalization requires maxmax and minmin")
                data = (data - minmin) / (maxmax - minmin)

            # smooth
            data = gaussian_filter(data, sigma=sigma)
        except Exception as e:
            print(f"Error processing well {well['ID'].iloc[0]}: {e}")
            return None, None, None
        
        return data, np.sort(df_sorted["CHK"].unique()), np.sort(df_sorted["WGL"].unique())
            
    experiment_dir = Path(f"{DATA_DIR}/{experiment_name}")
    runs = [r for r in experiment_dir.iterdir() if r.is_dir()]
    n_runs = len(runs)

    all_grids = []
    x_vals, y_vals = None, None

    for run_idx, run in enumerate(runs):
        print(f"Processing Run {run_idx}/{n_runs-1}...")
        path = Path(f"{run}/iteration_0/iteration_0.csv")
        if not path.exists():
            print(f"No valid data found for Run {run_idx} (file missing). Skipping.")
            continue

        df = pd.read_csv(path)
        wells_grouped = df.groupby("ID")

        # global normalization reference
        maxmax = df["WOIL"].max()
        minmin = df["WOIL"].min()

        # select wells
        wells_to_plot = wells or list(wells_grouped.groups.keys())

        for well_idx in wells_to_plot:
            well = wells_grouped.get_group(well_idx)
            grid, x_vals, y_vals = compute_normalized_grid(
                well, sigma=sigma, normalize=normalize, maxmax=maxmax, minmin=minmin
            )
            if grid is None:
                continue
            all_grids.append(grid)

    if not all_grids:
        print("No valid wells found for averaging.")
        return

    x_vals = np.linspace(0, 1, 50)
    y_vals = np.linspace(0, 5, 50)
    # Ensure all grids have the same shape
    shapes = [g.shape for g in all_grids]
    common_shape = max(set(shapes), key=shapes.count)  # most frequent shape

    filtered_grids = [g for g in all_grids if g.shape == common_shape]
    if len(filtered_grids) < len(all_grids):
        print(f"Skipping {len(all_grids) - len(filtered_grids)} grids due to shape mismatch.")


    # --- Compute mean grid ---
    all_grids = np.stack(filtered_grids, axis=0)
    mean_grid = np.nanmean(all_grids, axis=0)

    with plt.rc_context(mpl.rcParamsDefault):
        with plt.rc_context(CUSTOM_RC):

            # --- Plot mean contour ---
            plt.figure(figsize=(6, 6))
            contour = plt.contourf(
                x_vals,
                y_vals,
                mean_grid,
                levels=60,
                cmap="viridis",
                vmin=0,
                vmax=1
            )
            # plt.colorbar(contour, label="Average Normalized WOIL")
            # plt.title(f"Average Functional Landscape\n({experiment_name})")
            plt.xlabel("Choke")
            plt.ylabel("Gas lift")
            plt.tight_layout()
            if save:
                save_dir = f"{PLOT_DIR}/{experiment_name}"
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(f"{save_dir}/mean_landscape_{'_'.join(objective)}.png", dpi=300, bbox_inches="tight")
            plt.show()
    
def plot_penalty_terms(experiments: list[Path], 
                       iterations: int = 50,
                       only_optimizing_iterations: bool = True,
                       save: bool = False):
    """
    Plots the value of the penalty term across multiple experiments.
    Each experiment is averaged over its runs.
    """
    def calculate_penalty(water_prod, gradient: SPSAGradient, only_optimizing: bool = True):
        penalties = []
        if not only_optimizing:
            for i in range(len(water_prod)):
                violation = gradient.constraints.get_violations({"water": water_prod[i]})
                penalty = gradient._compute_penalty(violation)
                penalties.append(penalty)
        else:
            for i in range(2, len(water_prod), 3):
                violation = gradient.constraints.get_violations({"water": water_prod[i]})
                penalty = gradient._compute_penalty(violation)
                penalties.append(penalty)

        return penalties

    def calculate_lagragrian_term(water_prod, gradient: SPSAGradient, b: float, beta: float, only_optimizing: bool = True):
        lagr_penalties = []
        for i in range(0, len(water_prod), 3):
            gradient.bk = b / (1 + (i//3))**beta  # Update bk based on iteration
            if not only_optimizing:
                lagr_penalties.append(gradient._compute_lagrangian(
                                        gradient.constraints.get_violations({"water": water_prod[i]})))
                lagr_penalties.append(gradient._compute_lagrangian(
                                        gradient.constraints.get_violations({"water": water_prod[i+1]})))
            lagr_penalties.append(gradient._compute_lagrangian(
                                    gradient.constraints.get_violations({"water": water_prod[i+2]})))  # Only consider the resulting state
            
            # Update multipliers after each full iteration
            gradient.update_lambdas({"water": water_prod[i]}, {"water": water_prod[i+1]})
        return lagr_penalties

    fig, ax = plt.subplots(figsize=(10, 6))

    for experiment_dir in experiments:
        info = extract_settings(experiment_dir)
        config_file = info["config_file"]
        info.update(INIT_INFO[config_file])

        constraints = WellSystemConstraints(**info["constraints"])
        gradient = SPSAGradient(constraints=constraints,
                                use_penalty=True if info["rho"] > 0.0 else False,
                                use_lagrangian=True if info["b"] > 0.0 else False,
                                rho=info["rho"],
                                bk=info["b"])

        runs = [r for r in experiment_dir.iterdir() if r.is_dir()]
        n_runs = len(runs)

        penalties = []
        lagrangians = []
        all_terms = []
        for run_idx, run in enumerate(runs):
            print(f"Processing Run {run_idx}/{n_runs-1}...")
            path = Path(f"{run}/iteration_{iterations}/iteration_{iterations}.csv")
            if not path.exists():
                print(f"No valid data found for Run {run_idx} (file missing). Skipping.")
                continue

            df = pd.read_csv(path)
            n_sims = iterations

            oil, gasl, water = extract_production_history(data=df, 
                                                        n_sims=n_sims, 
                                                        init_production=(info["oil"], info["gaslift"], info["water"]),
                                                        only_optimizing=False)

            # Compute penalty terms for each run
            water = water[1:] # Exclude initial production

            penalty = calculate_penalty(water, copy.deepcopy(gradient), only_optimizing=only_optimizing_iterations)
            penalties.append(penalty)
            
            if gradient.use_lagrangian:
                lagrangian = calculate_lagragrian_term(water, 
                                                   copy.deepcopy(gradient), 
                                                   b=info["b"], 
                                                   beta=info["beta"], 
                                                   only_optimizing=only_optimizing_iterations)
            
                lagrangians.append(lagrangian)

        all_penalties = np.array(penalties)
        mean_penalties = np.mean(all_penalties, axis=0)
        if gradient.use_lagrangian:
            all_lagrangians = np.array(lagrangians)
            mean_lagrangians = np.mean(all_lagrangians, axis=0)

        if gradient.use_lagrangian:
            all_terms = [p + l for p, l in zip(mean_penalties, mean_lagrangians)]

        ax.plot(mean_penalties, label=experiment_dir.name)
        if gradient.use_lagrangian:
            ax.plot(mean_lagrangians, label=f"{experiment_dir.name} (Lagrangian)", linestyle='--')
            ax.plot(all_terms, label=f"{experiment_dir.name} (Total)", linestyle=':')

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Penalty Term")
    ax.set_title("Penalty Terms Across Experiments")
    ax.legend()
    plt.tight_layout()

    if save:
        save_dir = f"{PLOT_DIR}/penalty_terms"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/penalty_terms.png", dpi=300, bbox_inches="tight")

    plt.show()

def print_production_evolution(main_path: str, well_tbh: float | str):
    """
    For each subexperiment inside a main experiment folder, print the average
    initial and final production for a specific well (identified by its TBH)
    across all runs. The output is a table with subexperiments as rows and the
    six requested production metrics as columns.
    """
    subexperiments = sorted([p for p in main_path.iterdir() if p.is_dir()])
    if not subexperiments:
        print(f"No subexperiments found in {main_path}")
        return

    try:
        target_tbh = float(well_tbh)
    except (TypeError, ValueError):
        target_tbh = None

    def _select_well(df: pd.DataFrame) -> pd.DataFrame:
        """Return rows matching the requested TBH with numeric fallback."""
        if "TBH" not in df.columns:
            raise ValueError("TBH column not found in data.")
        
        well_df = df.loc[(df["TBH"].round(3) == round(target_tbh, 3))]
        return well_df

    def _iteration_number(path: Path) -> int | None:
        match = re.search(r"iteration_(\d+)$", path.parent.name)
        return int(match.group(1)) if match else None

    table_rows: list[dict] = []

    for subexp in subexperiments:
        init_vals = []
        final_vals = []
        run_dirs = sorted([r for r in subexp.iterdir() if r.is_dir()])

        for run in run_dirs:
            iter_paths = [
                p for p in run.glob("iteration_*/iteration_*.csv")
                if p.stem == p.parent.name and _iteration_number(p) is not None
            ]
            if not iter_paths:
                continue

            iter_paths = sorted(iter_paths, key=lambda p: _iteration_number(p))
            file = iter_paths[-1]

            file = pd.read_csv(file)
            well_data = _select_well(file)

            if well_data.empty:
                continue

            init_row = well_data.iloc[0]
            final_row = well_data.iloc[-1]

            try:
                init_vals.append((float(init_row["WOIL"]), float(init_row["WGL"]), float(init_row["WWAT"])))
                final_vals.append((float(final_row["WOIL"]), float(final_row["WGL"]), float(final_row["WWAT"])))
            except KeyError as e:
                raise KeyError(f"Expected production column missing: {e}") from e

        if not init_vals or not final_vals:
            continue

        init_mean = np.mean(init_vals, axis=0)
        final_mean = np.mean(final_vals, axis=0)

        table_rows.append({
            "Subexperiment": subexp.name,
            "Init WOIL": init_mean[0],
            "Final WOIL": final_mean[0],
            "Init WGL": init_mean[1],
            "Final WGL": final_mean[1],
            "Init WWAT": init_mean[2],
            "Final WWAT": final_mean[2],
        })

    if not table_rows:
        print(f"No production data found for well TBH={well_tbh} in {main_path}")
        return

    table_df = pd.DataFrame(table_rows, columns=[
        "Subexperiment", "Init WOIL", "Final WOIL", "Init WGL", "Final WGL", "Init WWAT", "Final WWAT"
    ])
    fmt = lambda x: "N/A" if pd.isna(x) else f"{x:.3f}"
    formatters = {col: fmt for col in table_df.columns if col != "Subexperiment"}

    print(table_df.to_string(index=False, formatters=formatters, col_space=12))

if __name__ == "__main__":
    # plot_spsa_experiment(experiment_name="experiments maxwells/20wells_perturb10", only_optimizing_iterations=True)
    # plot_decision_vector(experiment_name="experiments fixed gradient gain sequence/rho4_water20")
    # plot_decision_vector_series(experiment_name="experiments rho v3/rho2_water20")
    # print_production_sequence(experiment_name="experiments cyclicSPSA/20wells_perturb8")
    # plot_decision_vector_history(experiment_name="experiments rho v3/rho8_water20", wells_to_plot=None, only_optimizing_iterations=True, runs=None, type="scatter", save=False)
    # plot_step_size(experiment_name="experiments rho v3/rho8_water20", n_runs=10, iteration=50, save=True)
    # plot_multiple_function_landscapes(experiment_name="grid evaluation", wells=[2,7,11,13,25,37,8], sigma=1.0, normalize="local", objective=["WWAT"], save=True)
    # plot_mean_function_landscape(experiment_name="grid evaluation", wells=None, sigma=1.0, normalize="local", objective=["WWAT"], save=True)


    # ======= Run this if you want to see a set of experiments within a main folder =======
    # main_exp = "experiments rho final" # Change this as needed
    # main_exp = "experiments gl constraints"
    # main_exp = "experiments maxwells"
    main_exp = "experiments auglagrangian"
    # main_exp = "experiments rho max stepsize"
    # main_exp = "experiments fixed gradient gain sequence"
    # main_exp = "experiments optchoke"
    # main_exp = "experiments scaling factor"
    # main_exp = "experiments cyclicSPSA"

    main_path = Path(f"{os.environ['RESULTS_DIR']}/{main_exp}")
    # experiments = [e for e in main_path.iterdir() if e.is_dir() and "water20" in e.name]

    # for exp in experiments:
    #     plot_spsa_experiment(experiment_name=f"{main_exp}/{exp.name}", only_optimizing_iterations=True, save=False)
    #     plot_decision_vector(experiment_name=f"{main_exp}/{exp.name}", save=False, iteration=None)
        # plot_decision_vector_series(experiment_name=f"{main_exp}/{exp.name}", save_each=False, start=None, stop=None)
        # plot_decision_vector_history(experiment_name=f"{main_exp}/{exp.name}", wells_to_plot=None, only_optimizing_iterations=True, runs=None, type="scatter", save=False)
        # plot_decision_vector_history(experiment_name=f"{main_exp}/{exp.name}", wells_to_plot=None, only_optimizing_iterations=True, runs=None, type="line", save=False)
        # plot_step_size(experiment_name=f"{main_exp}/{exp.name}", n_runs=None, iteration=None, save=False)
    

    # Average production across experiments in a main folder
    # experiments = [e for e in main_path.iterdir() if (e.is_dir() and "20" in e.name)]
    # plot_average_production(experiments=experiments, only_optimizing_iterations=True, production_types=["oil", "water"], save=True)

    # Compare penalty terms across experiments in different main experiments
    main_experiments = [
        "experiments auglagrangian",
        "experiments rho final",
    ]
    main_paths = [Path(f"{os.environ['RESULTS_DIR']}/{me}") for me in main_experiments]
    experiments = []
    for main_path in main_paths:
        exps = [e for e in main_path.iterdir() if e.is_dir() and "rho1" in e.name and "water20" in e.name]
        experiments.extend(exps)
    plot_penalty_terms(experiments)