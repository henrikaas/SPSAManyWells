"""
Dataclass for holding constraints related to the well system.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
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
    u_max: float = 1.0
    u_min: float = 0.0

    # Gas lift
    gl_max: float
    gl_min: float = 0.0
    comb_gl_max: float 

    # Water in separator
    wat_max: float

    # # Constraints on move lengths
    # r_max: float = 1.0
    # r_min: float = 0.0

    # # Number of moves
    # max_moves: int
    # max_wells: int

    def __post_init__(self):
        # Placeholder for well-coupling constraints
        self.coupling_constraints: dict[str, float] = {"water": self.wat_max}

    def get_violations(self, y):
        """
        Compute the constraint violations based on the current state, `y`.

        Args:
            y (dict): Dictionary containing current state values.

        Returns:
            np.ndarray: Array of constraint violations. Positive values indicate violations.
        """
        violations = []

        for constraint, limit in self.coupling_constraints.items():
            violation = y[constraint] - limit
            violations.append(violation)

        return np.array(violations)

