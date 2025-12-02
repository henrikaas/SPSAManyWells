"""
Main logic for implementing Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm on gas-lifted petroleum production. 
"""
import numpy as np
import pandas as pd
import copy
import os
import multiprocessing as mp
from dataclasses import dataclass, replace
import random

from manywells.simulator import SSDFSimulator, SimError, WellProperties, BoundaryConditions
from scripts.data_generation.well import Well
import manywells.pvt as pvt

from spsa.constraints import WellSystemConstraints
from spsa.gradient import SPSAGradient

from spsa.utils import create_data_point, configure_wells, create_sim_results_df, calculate_state, create_dirs
from spsa.utils import save_data, choked_flow, save_fail_log, append_fail_log, save_init_log


@dataclass(frozen=True)
class SPSAConfig:
    a: float = 0.1          # learning-rate controller
    b: float = 0.0          # dual-rate controller NB: Needs to be 0 to disable Lagrangian
    c: float = 0.15         # perturbation magnitude
    A: int   = 5            # stabilizer
    alpha: float = 0.602    # learning-rate decay
    beta:  float = 0.602    # dual-rate decay
    gamma: float = 0.051    # perturbation decay
    sigma: float = 0.0      # noise level for oil evaluation
    rho:   float = 8.0      # penalty parameter for penalty method. NB: Needs to be 0 to disable penalty method

    def validate(self) -> "SPSAConfig":
        if not (0 <= self.alpha <= 1):
            raise ValueError("alpha must be in [0, 1]")
        if not (0 <= self.beta  <= 1):
            raise ValueError("beta must be in [0, 1]")
        if not (0 <= self.gamma <= 1):
            raise ValueError("gamma must be in [0, 1]")
        if self.a <= 0 or self.c <= 0 or self.A < 0:
            raise ValueError("a, c must be > 0 and A >= 0")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")
        return self

BERNOULLI_DIRECTIONS = [-1, 1]
HYPERPARAM_PRESETS: dict[str, SPSAConfig] = {
    "default": SPSAConfig(),
    "slower_ak": SPSAConfig(a=0.15, A=5, alpha=0.502),
    # TODO: Tune these presets
    # "robust":  SPSAConfig(a=0.1, c=0.2, A=100, alpha=0.2, beta=0.5, sigma=0.1, rho=0.1),
    # "fast":    SPSAConfig(a=0.5, c=0.05, A=10,  alpha=0.4, beta=0.7, sigma=0.0, rho=0.0),
}
CONSTRAINT_PRESETS: dict[str, WellSystemConstraints] = {
    "default": WellSystemConstraints(gl_max=5.0, comb_gl_max=10.0, wat_max=20.0, max_wells=5),
    "strict_water": WellSystemConstraints(gl_max=5.0, comb_gl_max=10.0, wat_max=10.0, max_wells=5),
    "a_bit_strict_water": WellSystemConstraints(gl_max=5.0, comb_gl_max=10.0, wat_max=15.0, max_wells=5),
    "relaxed": WellSystemConstraints(gl_max=1000, comb_gl_max=1000, wat_max=1000, max_wells=1000),
    "relaxed_water": WellSystemConstraints(gl_max=5.0, comb_gl_max=10.0, wat_max=1000, max_wells=5),
    # TODO: Define more presets
    # "strict":  WellSystemConstraints(gl_max=100, comb_gl_max=200, wat_max=300, max_wells=5),
}

class SPSA:
    """
    Implementation of SPSA optimization algorithm
    """

    def __init__(self, 
                 wells: list[Well], 
                 constraints: WellSystemConstraints | None = None,
                 *,
                 hyperparam_config: SPSAConfig | None = None,
                 hyperparam_preset: str | None = None,
                 **hyperparam_overrides,  # When passing individual hyperparameters
                 ):

        self.wells: list[Well] = wells
        self.n_wells = len(wells)
        self.backup_wells = copy.deepcopy(wells) # Backup of the last successful well states

        # Hyperparameter configuration
        # Choose base config: explicit > preset > default
        base = hyperparam_config or (HYPERPARAM_PRESETS[hyperparam_preset] if hyperparam_preset else HYPERPARAM_PRESETS["default"])
        # Apply overrides
        for k in hyperparam_overrides:
            if not hasattr(base, k):
                raise TypeError(f"Unknown hyperparameter: {k}")
        cfg = replace(base, **hyperparam_overrides).validate()

        self.hyperparams = cfg # Hyperparameter configuration

        # Initialize constraint object
        if constraints is None:
            print(f"No constraints provided. Using default constraints.")
            constraints = CONSTRAINT_PRESETS["default"]
        if constraints.max_wells > self.n_wells:
            print(f"Warning: max_wells ({constraints.max_wells}) > number of wells ({self.n_wells}). Setting max_wells = {self.n_wells}.")
            constraints = replace(constraints, max_wells=self.n_wells)
        self.constraints: WellSystemConstraints = constraints.validate()

        self.scaling_factor = min(10, self.constraints.gl_max) # Max value to avoid too large perturbations when gl_max constraint is relaxed
        self.use_cyclic = True if self.constraints.max_wells < self.n_wells else False # Use cyclic SPSA if max_wells constraint is active

        # Initialize gradient object
        use_penalty = True if self.hyperparams.rho > 0 else False
        use_lagrangian = True if self.hyperparams.b > 0 else False
        print(f"Initializing SPSA with the following SPSA gradient settings:")
        print(f"  Use Penalty: {use_penalty}")
        print(f"  Use Lagrangian: {use_lagrangian}")

        self.gradient = SPSAGradient(constraints=self.constraints,
                                     use_penalty=use_penalty,
                                     rho=self.hyperparams.rho,
                                     use_lagrangian=use_lagrangian,
                                     bk=self.hyperparams.b  # Needed when using lagrangian
                                     )
        
        
    def _draw_directions(self) -> tuple[int]:
        """
        Randomly draws directions for 'u' and 'gl' from the set of Bernoulli directions.

        Returns:
            tuple[int]: A tuple containing two integers, each representing a randomly chosen direction
                        for 'u' and 'gl', respectively, from BERNOULLI_DIRECTIONS.
        """
        u_direction = np.random.choice(BERNOULLI_DIRECTIONS)
        gl_direction = np.random.choice(BERNOULLI_DIRECTIONS)
        return (u_direction, gl_direction)

    def _draw_subvector(self) -> list[list[int]]:
        """
        Draws a set of subvectors for cyclic SPSA.
        Each subvector is a set of well indices that will be perturbed together in each iteration.
        The size of each subvector is <= max_wells.

        Returns:
            list[list[int]]: A list of subvectors, where each subvector is a list of well indices.
        """
        max_wells = self.constraints.max_wells

        indices = np.arange(self.n_wells)
        np.random.shuffle(indices)
        subvectors = [indices[i:i + max_wells].tolist() for i in range(0, self.n_wells, max_wells)]
        return subvectors
    
    def _generate_perturbation(self, ck: float, u: float, gl: float, has_gl: bool):
        """
        Generates positive and negative perturbations for the SPSA algorithm.
        This method computes two perturbed parameter vectors (positive and negative) based on the current values of `u` and `gl`, a perturbation scale `ck`, and random directions. The perturbations are clipped to remain within specified constraints.
        Args:
            ck (float): The perturbation scale factor.
            u (float): The current value of the 'u' parameter.
            gl (float): The current value of the 'gl' parameter.
            has_gl (bool): Flag indicating whether the 'gl' parameter should be perturbed.
        Returns:
            Tuple[List[np.ndarray], List[float]]:
                - A list containing the positive and negative perturbed parameter vectors.
                - A list containing the random directions used for 'u' and 'gl' perturbations.
        """
        u_direction, gl_direction = self._draw_directions()
        gl_direction = gl_direction if has_gl else 0.0

        pos_perturbation = [u + ck * u_direction,
                            gl + ck * gl_direction * self.scaling_factor]
        neg_perturbation = [u - ck * u_direction,
                            gl - ck * gl_direction * self.scaling_factor]
        
        pos_perturbation = self.constraints.enforce_well_constraints(pos_perturbation)
        neg_perturbation = self.constraints.enforce_well_constraints(neg_perturbation)

        return [pos_perturbation, neg_perturbation], [u_direction, gl_direction]
    
    def _single_simulation(self, simulator: SSDFSimulator, well: Well):

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
    
    def _parallel_simulation(self, well_idx: int, well: Well, sim: SSDFSimulator):
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
            x = self._single_simulation(simulator=sim, well=well)
        except SimError:
            return well_idx, None

        return well_idx, x
    
    def optimize(self, n_sim: int = 50, starting_k: int = 0, save_path: str = None):
        """
        Run the SPSA optimization algorithm.
        Args:
            n_sim (int): Number of simulations to run.
            starting_k (int): Starting iteration number (useful for resuming).
            save_path (str): Path to save the results. If None, results are not saved.
        """
        n_sim_wells = self.constraints.max_wells
        if self.use_cyclic:
            pert_vectors = self._draw_subvector() # Draw the subvectors for cyclic SPSA
            subvector_k = [0 for _ in range(len(pert_vectors))] # Track iteration number for each subvector, used for the a_k step size calculation
        else:
            pert_vectors = [list(range(self.n_wells))] # Single vector with all wells

        well_idxs = np.arange(0, self.n_wells)

        simulators = [SSDFSimulator(w.wp, w.bc) for w in self.wells] # Simulator object for each well

        # Create data point storage
        well_data = [create_sim_results_df() for _ in range(self.n_wells)]

        n_fails = 0
        fails_per_well: dict[float, list] = {
            well.wp.L: [] for well in self.wells
        } # Track number of failed simulations per well

        k = starting_k + 1
        while k <= n_sim:
            if n_fails >= 10:
                save_fail_log(path=save_path, k=k-1, fails_per_well=fails_per_well, success=False)
                raise SimError('Too many fails. Exiting SPSA...')

            # ========== 1 ==============
            # Initialize each iteration
            # Update SPSA parametres, init placeholders and sample new well conditions
            # ===========================
            # Calculate SPSA parametres
            ak = self.hyperparams.a / ((k + self.hyperparams.A) ** self.hyperparams.alpha)
            self.gradient.bk = self.hyperparams.b / (k**self.hyperparams.beta)
            ck = self.hyperparams.c / (k ** self.hyperparams.gamma)

            staged_data = [[] for _ in range(self.n_wells)] # Placeholder for data in current iteration
            pos_well_data = [] # Reset placeholder for simulation results in positive perturbation
            neg_well_data = [] # Reset placeholder for simulation results in negative perturbation
            stat_well_data = [] # Reset placeholder for simulation results in stationary wells

            # Sample new well conditions
            for idx, well in enumerate(self.wells):
                # well = well.sample_new_conditions(sample_u=False, sample_w_lg=False)
                self.wells[idx] = well
                # Update simulator with new well conditions
                simulators[idx].wp = well.wp
                simulators[idx].bc = well.bc

            # ============= 2 ==============
            # Perturb the system
            # Draw perturbation vector and simulate in each direction
            # ==============================
            chosen_wells_idxs = random.choice(pert_vectors)
            unselected_well_idxs = list(set(well_idxs) - set(chosen_wells_idxs))
            cur_state = np.array([[self.wells[idx].bc.u, self.wells[idx].bc.w_lg] for idx in chosen_wells_idxs]) # Current state of the decision vector in the chosen wells

            # Calculate the perturbation vectors
            perturbations = []
            directions = []
            for idx, well_idx in enumerate(chosen_wells_idxs):
                p_vector, p_direction = self._generate_perturbation(ck=ck,
                                     u = cur_state[idx][0],
                                     gl = cur_state[idx][1],
                                     has_gl=self.wells[well_idx].has_gas_lift)
                
                perturbations.append(p_vector)
                directions.append(p_direction)
            
            # Project gas lift values to satisfy constraints
            perturbations = np.array(perturbations)
            stat_gl = sum(self.wells[idx].bc.w_lg for idx in unselected_well_idxs) # Gas lift used by stationary wells
            perturbations[:, 0, 1] = self.constraints.project_combined_gl(perturbations[:, 0, 1], stat_gl=stat_gl) # Project combined gas lift for positive perturbation
            perturbations[:, 1, 1] = self.constraints.project_combined_gl(perturbations[:, 1, 1], stat_gl=stat_gl) # Project combined gas lift for negative perturbation

            try:
                # TODO: Refactor the perturbation into a separate function?
                # Positive perturbation
                tasks = []
                for idx, well_idx in enumerate(chosen_wells_idxs):
                    self.wells[well_idx].bc.u = perturbations[idx][0][0]
                    self.wells[well_idx].bc.w_lg = perturbations[idx][0][1]

                    tasks.append((
                        well_idx,
                        self.wells[well_idx],
                        simulators[well_idx],
                    ))

                with mp.Pool(processes=min(mp.cpu_count()-2, n_sim_wells)) as pool:
                    results = pool.starmap(self._parallel_simulation, tasks)
                
                for well_idx, x in results:
                    well = self.wells[well_idx]

                    if x is None: # Indicates failed simulation
                        key = self.wells[well_idx].wp.L
                        fails_per_well[key] = append_fail_log(fails_per_well[key], self.wells[well_idx], k)
                        raise SimError(f"Simulation failed for well {well_idx}")
                    else:
                        well.x_guesses.append(x) # Store the result as a guess if simulation was successful
                        dp = create_data_point(well=well, sim=simulators[well_idx], x=x, sim_type='Pos. Perturbation')

                    pos_well_data.append(dp)
                    staged_data[well_idx].append(dp)
                
                # Negative perturbation
                tasks = []
                for idx, well_idx in enumerate(chosen_wells_idxs):
                    self.wells[well_idx].bc.u = perturbations[idx][1][0]
                    self.wells[well_idx].bc.w_lg = perturbations[idx][1][1]

                    tasks.append((
                        well_idx,
                        self.wells[well_idx],
                        simulators[well_idx],
                    ))

                with mp.Pool(processes=min(mp.cpu_count(), n_sim_wells, 8)) as pool:
                    results = pool.starmap(self._parallel_simulation, tasks)

                for well_idx, x in results:
                    well = self.wells[well_idx]

                    if x is None: # Indicates failed simulation
                        key = self.wells[well_idx].wp.L
                        fails_per_well[key] = append_fail_log(fails_per_well[key], self.wells[well_idx], k)
                        raise SimError(f"Simulation failed for well {well_idx}")
                    else:
                        well.x_guesses.append(x)  # Store the result as a guess if simulation was successful
                        dp = create_data_point(well=well, sim=simulators[well_idx], x=x, sim_type='Neg. Perturbation')

                    neg_well_data.append(dp)
                    staged_data[well_idx].append(dp)

                # ============= 3 ==============
                # Simulate the wells that are not perturbed
                # ==============================
                #TODO: If we move this first, we dont need pos_well_data and neg_well_data

                tasks = []
                for well_idx in unselected_well_idxs:
                    tasks.append((
                        well_idx,
                        self.wells[well_idx],
                        simulators[well_idx],
                    ))
                if len(tasks) > 0:
                    with mp.Pool(processes=min(mp.cpu_count()-2, self.n_wells - n_sim_wells)) as pool:
                        results = pool.starmap(self._parallel_simulation, tasks)
                else:
                    results = []

                for well_idx, x in results:
                    well = self.wells[well_idx]

                    if x is None: # Indicates failed simulation
                        key = self.wells[well_idx].wp.L
                        fails_per_well[key] = append_fail_log(fails_per_well[key], self.wells[well_idx], k)
                        raise SimError(f"Simulation failed for well {well_idx}")
                    else:
                        well.x_guesses.append(x)  # Store the result as a guess if simulation was successful
                        dp = create_data_point(well=well, sim=simulators[well_idx], x=x, sim_type='Unselected Well')

                    stat_well_data.append(dp)
                    staged_data[well_idx].append(dp)

                # ============= 4 ==============
                # Calculate the state and gradient of the system in both perturbations
                # ==============================
                y_pos = calculate_state(well_data=pos_well_data + stat_well_data)
                y_neg = calculate_state(well_data=neg_well_data + stat_well_data)

                if self.use_cyclic:
                    # Update subvector iteration count
                    subvector_idx = pert_vectors.index(chosen_wells_idxs)
                    subvector_k[subvector_idx] += 1
                    ak = self.hyperparams.a / ((subvector_k[subvector_idx] + self.hyperparams.A) ** self.hyperparams.alpha)

                # Compute the gradient
                gradient = self.gradient.compute_gradient(y_pos=y_pos, y_neg=y_neg, delta=np.array(directions))
                step_size = gradient * ak
                np.clip(step_size, -0.2, 0.2, out=step_size) # Clip step size to avoid too large steps
                step_size[:,1] *=  self.scaling_factor # Scale the gradient for gas lift wells

                # Update decision variables
                opt_theta = cur_state - step_size
                opt_theta = self.constraints.enforce_well_constraints(opt_theta) # Project decision vector back to legal space
                opt_theta[:,1] = self.constraints.project_combined_gl(opt_theta[:,1], stat_gl=stat_gl) # Project gas lift values to satisfy combined constraint

                # ============= 5 ==============
                # Update the well states with the optimized decision variables and run optimizing simulations
                # ==============================
                # Run optimizing simulation
                tasks = []
                for idx, well_idx in enumerate(chosen_wells_idxs):
                    self.wells[well_idx].bc.u = opt_theta[idx][0]
                    self.wells[well_idx].bc.w_lg = opt_theta[idx][1]

                    tasks.append((
                        well_idx,
                        self.wells[well_idx],
                        simulators[well_idx],
                    ))

                with mp.Pool(processes=min(mp.cpu_count()-2, n_sim_wells)) as pool:
                    results = pool.starmap(self._parallel_simulation, tasks)

                for well_idx, x in results:
                    well = self.wells[well_idx]

                    if x is None: # Indicates failed simulation
                        key = self.wells[well_idx].wp.L
                        fails_per_well[key] = append_fail_log(fails_per_well[key], self.wells[well_idx], k)
                        raise SimError(f"Simulation failed for well {well_idx}")
                    else:
                        well.x_guesses.append(x)  # Store the result as a guess if simulation was successful
                        dp = create_data_point(well=well, sim=simulators[well_idx], x=x, sim_type='Optimizing')

                    staged_data[well_idx].append(dp)

            except SimError as e:
                n_fails += 1
                self.wells = copy.deepcopy(self.backup_wells) # Reset wells to the last successful state
                print(f"Simulation failed: {e}")
                print(f"Number of failed simulations: {n_fails}")
                continue
            
            # ============= 6 ==============
            # If success, save data and update backup
            # ==============================
            # Update the Lagrangian multipliers
            self.gradient.update_lambdas(y_pos=y_pos, y_neg=y_neg)

            print("--------------------------------------------")
            print(f"Simulation #{k} successful.")
            print("--------------------------------------------")

            # Save data stream to main storage
            for i in range(self.n_wells):
                if staged_data[i]:  
                    well_data[i] = pd.concat([well_data[i], *staged_data[i]], ignore_index=True)
            y_opt = calculate_state(well_data=well_data)

            if k % 10 == 0 and save_path is not None:
                print(f"Saving state after {k} successful iterations")
                save_data(self.wells, well_data=well_data, main_path=save_path, k=k)

            self.backup_wells = copy.deepcopy(self.wells) # Back up of wells for next iteration
            k += 1
            n_fails = 0

        if save_path is not None:
            print(f"Saving final state after {k-1} successful iterations")
            if (k - 1) % 10 != 0:
                # Save data stream
                save_data(self.wells, well_data=well_data, main_path=save_path, k=k-1)
            # Save failure log
            save_fail_log(path=save_path, k=k-1, fails_per_well=fails_per_well, success=True)

        print("---------------------------------------------")
        print("SPSA successful.")
        print("Final hyperparametres:")
        # print(f"Lambdas: {self.lambdas}")


if __name__ == "__main__":
    n_runs = 20
    n_sim = 100

    experiments = [
        # Experiment on using cyclic SPSA on a 20well system
        # Max wells = 2
        {"config": "20randomwells",
         "save": "experiments cyclicSPSA/20wells_perturb2",
         "description": "Experiment on cyclicSPSA for different values of max wells\n"
                        "Clips the gradient step size to max 0.2\n"
                        "No new sampling of conditions!\n"
                        "maxwells = 4\n"
                        "20random wells\n",
         "start": "choke 0.5 | Gas lift 0.0",
         "n_wells": 20,
         "constraints": WellSystemConstraints(gl_max=5.0, comb_gl_max=15.0, wat_max=225, max_wells=2),
         "hyperparams": HYPERPARAM_PRESETS["slower_ak"],
         "hyperparam_overrides": {},
        },
        # Max wells = 3
        {"config": "20randomwells",
         "save": "experiments cyclicSPSA/20wells_perturb3",
         "description": "Experiment on cyclicSPSA for different values of max wells\n"
                        "Clips the gradient step size to max 0.2\n"
                        "No new sampling of conditions!\n"
                        "maxwells = 3\n"
                        "20random wells\n",
         "start": "choke 0.5 | Gas lift 0.0",
         "n_wells": 20,
         "constraints": WellSystemConstraints(gl_max=5.0, comb_gl_max=15.0, wat_max=225, max_wells=3),
         "hyperparams": HYPERPARAM_PRESETS["slower_ak"],
         "hyperparam_overrides": {},
        },
        # Max wells = 4
        {"config": "20randomwells",
         "save": "experiments cyclicSPSA/20wells_perturb4",
         "description": "Experiment on cyclicSPSA for different values of max wells\n"
                        "Clips the gradient step size to max 0.2\n"
                        "No new sampling of conditions!\n"
                        "maxwells = 4\n"
                        "20random wells\n",
         "start": "choke 0.5 | Gas lift 0.0",
         "n_wells": 20,
         "constraints": WellSystemConstraints(gl_max=5.0, comb_gl_max=15.0, wat_max=225, max_wells=4),
         "hyperparams": HYPERPARAM_PRESETS["slower_ak"],
         "hyperparam_overrides": {},
        },
        # Max wells = 5
        {"config": "20randomwells",
         "save": "experiments cyclicSPSA/20wells_perturb5",
         "description": "Experiment on cyclicSPSA for different values of max wells\n"
                        "Clips the gradient step size to max 0.2\n"
                        "No new sampling of conditions!\n"
                        "maxwells = 5\n"
                        "20random wells\n",
         "start": "choke 0.5 | Gas lift 0.0",
         "n_wells": 20,
         "constraints": WellSystemConstraints(gl_max=5.0, comb_gl_max=15.0, wat_max=225, max_wells=5),
         "hyperparams": HYPERPARAM_PRESETS["slower_ak"],
         "hyperparam_overrides": {},
        },
        # Max wells = 8
        {"config": "20randomwells",
         "save": "experiments cyclicSPSA/20wells_perturb8",
         "description": "Experiment on cyclicSPSA for different values of max wells\n"
                        "Clips the gradient step size to max 0.2\n"
                        "No new sampling of conditions!\n"
                        "maxwells = 8\n"
                        "20random wells\n",
         "start": "choke 0.5 | Gas lift 0.0",
         "n_wells": 20,
         "constraints": WellSystemConstraints(gl_max=5.0, comb_gl_max=15.0, wat_max=225, max_wells=8),
         "hyperparams": HYPERPARAM_PRESETS["slower_ak"],
         "hyperparam_overrides": {},
        },
        # Max wells = 10
        {"config": "20randomwells",
            "save": "experiments cyclicSPSA/20wells_perturb10",
            "description": "Experiment on cyclicSPSA for different values of max wells\n"
                            "Clips the gradient step size to max 0.2\n"
                            "No new sampling of conditions!\n"
                            "maxwells = 10\n"
                            "20random wells\n",
            "start": "choke 0.5 | Gas lift 0.0",
            "n_wells": 20,
            "constraints": WellSystemConstraints(gl_max=5.0, comb_gl_max=15.0, wat_max=225, max_wells=10),
            "hyperparams": HYPERPARAM_PRESETS["slower_ak"],
            "hyperparam_overrides": {},
            },
        # Max wells = 15
        {"config": "20randomwells",
            "save": "experiments cyclicSPSA/20wells_perturb15",
            "description": "Experiment on cyclicSPSA for different values of max wells\n"
                            "Clips the gradient step size to max 0.2\n"
                            "No new sampling of conditions!\n"
                            "maxwells = 15\n"
                            "20random wells\n",
            "start": "choke 0.5 | Gas lift 0.0",
            "n_wells": 20,
            "constraints": WellSystemConstraints(gl_max=5.0, comb_gl_max=15.0, wat_max=225, max_wells=15),
            "hyperparams": HYPERPARAM_PRESETS["slower_ak"],
            "hyperparam_overrides": {},
            },
        # Max wells = 20
        {"config": "20randomwells",
            "save": "experiments cyclicSPSA/20wells_perturb20",
            "description": "Experiment on cyclicSPSA for different values of max wells\n"
                            "Clips the gradient step size to max 0.2\n"
                            "No new sampling of conditions!\n"
                            "maxwells = 20\n"
                            "20random wells\n",
            "start": "choke 0.5 | Gas lift 0.0",
            "n_wells": 20,
            "constraints": WellSystemConstraints(gl_max=5.0, comb_gl_max=15.0, wat_max=225, max_wells=20),
            "hyperparams": HYPERPARAM_PRESETS["slower_ak"],
            "hyperparam_overrides": {},
            },
    ]

    # ----------- Main script -----------
    work_dir, results_dir = create_dirs(experiments, n_runs)

    for experiment in experiments:
        parent_wells = configure_wells(filepath=
                                    os.path.join(work_dir, "config files", f"{experiment['config']}.csv"))
        parent_constraints = experiment['constraints']
        parent_hyperparams = experiment['hyperparams']
        overrides = experiment.get("hyperparam_overrides", {})

        parent_spsaopt = SPSA(wells=parent_wells,
                              constraints=parent_constraints,
                              hyperparam_config=parent_hyperparams,
                              **overrides)
        
        # Create initial log
        experiment.update({
            "n_sim": n_sim,
            "n_wells": parent_spsaopt.n_wells,
            "hyperparams": parent_spsaopt.hyperparams,
            "constraints": parent_spsaopt.constraints,
        })
        save_init_log(os.path.join(results_dir, experiment['save']), description=experiment)
        
        for run_id in range(n_runs):
            spsaopt = copy.deepcopy(parent_spsaopt)
            
            try:
                spsaopt.optimize(n_sim=n_sim,
                                 save_path=os.path.join(results_dir, experiment['save'], f"run{run_id}")
                )
            except SimError as e:
                print(f"Simulation {run_id} failed: {e}")
                continue