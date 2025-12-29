import numpy as np
from constraints import WellSystemConstraints

class SPSAGradient:

    def __init__(self, constraints: WellSystemConstraints, 
                use_penalty: bool = True,
                use_lagrangian: bool = False,
                rho: float = 1.0,
                bk: float = 1.0):
        """
        Initializes the gradient computation object with specified constraint handling options.
        Args:
            constraints (WellSystemConstraints): The constraints object defining the well system's coupling constraints.
            use_penalty (bool, optional): Whether to use a penalty method for constraint handling. Defaults to True.
            use_lagrangian (bool, optional): Whether to use a Lagrangian method for constraint handling. Defaults to False.
            rho (float, optional): Penalty parameter used if `use_penalty` is True. Defaults to 1.0.
            bk (float, optional): Parameter for the Lagrangian method, used if `use_lagrangian` is True. Defaults to 1.0.
        Attributes:
            constraints (WellSystemConstraints): The well system constraints.
            use_penalty (bool): Indicates if penalty method is used.
            use_lagrangian (bool): Indicates if Lagrangian method is used.
            rho (float): Penalty parameter (if applicable).
            b (float): Lagrangian parameter (if applicable).
            lambdas (dict[str, float]): Lagrange multipliers for each coupling constraint (if applicable).
            gradient (np.ndarray): Stores the computed gradient value.
        """
        
        self.constraints: WellSystemConstraints = constraints
        self.use_penalty: bool = use_penalty
        self.use_lagrangian: bool = use_lagrangian

        if self.use_penalty:
            self.rho: float = rho # Penalty parameter

        if self.use_lagrangian:
            self.bk: float = bk
            self.lambdas: dict[str, float] = {key: 0.0 for key in self.constraints.penalty_constraints.keys()} # Initialize Lagrange multipliers

        self.gradient: np.ndarray = None

    def compute_gradient(self, y_pos, y_neg, delta, ck):
        """
        Compute the gradient using the SPSA formula.

        Args:
            y_pos (float): Objective function value at positive perturbation.
            y_neg (float): Objective function value at negative perturbation.
            delta (np.ndarray): Perturbation vector.

        Returns:
            Array: Computed gradient.
        """
        violations_pos = self.constraints.get_violations(y_pos)
        violations_neg = self.constraints.get_violations(y_neg)

        L_pos = -self._compute_objective(y_pos) + self._compute_penalty(violations_pos) + self._compute_lagrangian(violations_pos)
        L_neg = -self._compute_objective(y_neg) + self._compute_penalty(violations_neg) + self._compute_lagrangian(violations_neg)

        grad = (L_pos - L_neg).reshape(-1, 1) / (2 * delta)
        grad[np.isinf(grad)] = 0 # Set infinite values to 0

        self.gradient = grad
        return self.gradient
    
    def _compute_objective(self, y):
        """
        Compute the value of the original objective function.
        """
        return y["oil"]

    
    def _compute_penalty(self, violations):
        """
        Compute penalty term for constraint violations.
        """
        if not self.use_penalty:
            return 0.0 # No penalty applied
        
        penalty = (self.rho / 2) * np.linalg.norm(np.maximum(0, violations)) ** 2
        return penalty

    def _compute_lagrangian(self, violations):
        """
        Compute Lagrangian term for constraint violations.
        """
        if not self.use_lagrangian:
            return 0.0 # No Lagrangian applied
        lagrangian = sum(self.lambdas[key] * max(0, violations[i]) for i, key in enumerate(self.constraints.penalty_constraints.keys()))
        return lagrangian

    def update_lambdas(self, y_pos, y_neg):
        """
        Update Lagrange multipliers based on current violations.
        """
        if not self.use_lagrangian:
            return
        
        pos_violations = self.constraints.get_violations(y_pos)
        neg_violations = self.constraints.get_violations(y_neg)

        violations = np.maximum(np.array(pos_violations), np.array(neg_violations))

        v = np.maximum(0, violations)
        for i, key in enumerate(self.lambdas.keys()):
            self.lambdas[key] += self.bk * max(0, v[i])        

        
        


