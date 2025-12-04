"""
Dataclass for holding constraints related to the well system.
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class WellSystemConstraints:
    """
    Dataclass representing operational constraints for a well system.

    Attributes:
        u_max (float): Maximum choke value. Always set to 1
        u_min (float): Minimum choke value. Always set to 0
        gl_max (float): Maximum gas lift value for a single well.
        gl_min (float): Minimum gas lift value. Always set to 0.
        comb_gl_max (float): Maximum combined gas lift value for the well system. 
        wat_max (float): Maximum water allowed in separator.

        # The following constraints are commented out:
        # r_max (float): Maximum move length. Default is 1 (whole solution space).
        # r_min (float): Minimum move length. Default is 0.
        # max_moves (int): Maximum number of moves allowed.
        # max_wells (int): Maximum number of wells allowed.
    """

    # Choke
    u_min: float = field(default=0.0, init=False)
    u_max: float = field(default=1.0, init=False)

    # Gas lift
    gl_min: float = field(default=0.0, init=False)
    gl_max: float
    comb_gl_max: float

    # Water in separator
    wat_max: float

    # # Constraints on move lengths
    # r_max: float = 1.0
    # r_min: float = 0.0

    # # Number of moves
    # max_moves: int
    max_wells: int
    
    l_max: float | None = None
    l_min: float = 0.0


    # Coupling constraints that is enforced by penalty and lagrangian methods
    penalty_constraints: dict[str, float] = field(init=False)

    def __post_init__(self):
        # Placeholder for well-coupling penalty constraints
        object.__setattr__(self, "penalty_constraints", {"water": self.wat_max})

    def validate(self):
        # Enforce constants exactly
        if self.u_min != 0.0 or self.u_max != 1.0:
            raise ValueError("u_min must be 0.0 and u_max must be 1.0.")
        if self.gl_min != 0.0:
            raise ValueError("gl_min must be 0.0.")

        # Gas lift ranges
        if self.gl_max <= self.gl_min:
            raise ValueError(f"gl_max must be > gl_min, got {self.gl_max}.")
        if self.comb_gl_max < self.gl_max:
            raise ValueError(f"comb_gl_max={self.comb_gl_max} must be ≥ gl_max={self.gl_max}.")

        # Water
        if self.wat_max <= 0:
            raise ValueError(f"wat_max must be > 0, got {self.wat_max}.")
        
        # Movement lengths
        if self.l_min < 0:
            raise ValueError(f"g_min must be >= 0, got {self.l_min}.")
        if self.l_max <= self.l_min:
            raise ValueError(f"g_max must be > g_min, got g_max={self.l_max}, g_min={self.l_min}.")

        # Wells
        if self.max_wells <= 0:
            raise ValueError(f"max_wells must be > 0, got {self.max_wells}.")
        
        return self

    def get_violations(self, y):
        """
        Compute the constraint violations based on the current state, `y`.

        Args:
            y (dict): Dictionary containing current state values.

        Returns:
            np.ndarray: Array of constraint violations. Positive values indicate violations.
        """
        violations = []

        for constraint, limit in self.penalty_constraints.items():
            violation = y[constraint] - limit
            violations.append(violation)

        return np.array(violations)

    def project_combined_gl(self, gl_values: np.ndarray, stat_gl: float) -> np.ndarray:
        """
        Project gas lift values to satisfy individual and combined constraints.

        Args:
            gl_values (np.ndarray): Array of gas lift values for each well.

        Returns:
            np.ndarray: Projected gas lift values satisfying the constraints.
        """
        # Clip individual gas lift values to be within [gl_min, gl_max]
        # Should be done before calling this function, but just in case
        gl_values = np.clip(gl_values, self.gl_min, self.gl_max)

        # Check if the combined gas lift exceeds the maximum allowed
        allowed_gl = self.comb_gl_max - stat_gl # Need to account for stationary wells, as we should not change these values
        total_gl = np.sum(gl_values)
        if total_gl > allowed_gl:
            # Scale down the gas lift values proportionally
            scaling_factor = allowed_gl / total_gl
            projected_gl = gl_values * scaling_factor
        else:
            projected_gl = gl_values

        return projected_gl

    def enforce_well_constraints(self, decision_vector):
        """
        Enforces lower and upper bounds on the decision vector for well constraints.
        Args:
            decision_vector (np.ndarray): Array containing decision variables [u, w_lg].
        Returns:
            np.ndarray: Decision vector with enforced constraints.
        """
        return np.clip(decision_vector, [self.u_min, self.gl_min], [self.u_max, self.gl_max])
    
    def project_step_size(self, step_size: np.ndarray) -> np.ndarray:
        """
        Project the step size to ensure that it does not violate the constraints.

        Args:
            step_size (np.ndarray): Array of step sizes for each well.

        Returns:
            np.ndarray: Projected step sizes satisfying the constraints.
        """
        if self.l_max is not None:
            np.clip(step_size, -self.l_max, self.l_max, out=step_size) # Clip step size to avoid too large steps
        step_size[np.abs(step_size) < self.l_min] = self.l_min * np.sign(step_size[np.abs(step_size) < 0.01]) # Set minimum absolute value

        return step_size
