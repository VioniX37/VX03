"""
Universal OptimizationModel container.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.sparse as sp

from sovereign_opt.model.variable import Variable, VariableType
from sovereign_opt.model.constraint import Constraint, ConstraintSense
from sovereign_opt.model.objective import Objective, ObjectiveSense


class ModelValidationError(Exception):
    """Raised when model validation fails."""
    pass


@dataclass
class ModelMetadata:
    name: str = "OptimizationModel"
    problem_class: str = "LP"
    num_variables: int = 0
    num_constraints: int = 0
    num_nonzeros: int = 0
    density: float = 0.0
    num_continuous: int = 0
    num_integer: int = 0
    num_binary: int = 0
    num_quadratic_terms: int = 0
    min_coefficient: float = 0.0
    max_coefficient: float = 0.0


class OptimizationModel:
    """
    Universal optimization model representation for LP, MILP, QP, and MIQP.
    """

    def __init__(self, name: str = "OptimizationModel"):
        self.name: str = name
        self.variables: Dict[str, Variable] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.objective: Objective = Objective()
        self._var_order: List[str] = []
        self._con_order: List[str] = []

    def add_variable(
        self,
        name: str,
        lower_bound: float = 0.0,
        upper_bound: float = float("inf"),
        var_type: VariableType = VariableType.CONTINUOUS,
    ) -> Variable:
        """Add a decision variable to the model."""
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists in model.")
        var = Variable(
            name=name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            var_type=var_type,
            column_index=len(self._var_order),
        )
        self.variables[name] = var
        self._var_order.append(name)
        return var

    def add_constraint(
        self,
        name: str,
        coefficients: Dict[str, float],
        sense: ConstraintSense = ConstraintSense.LE,
        rhs: float = 0.0,
        lower_bound: float = float("-inf"),
        upper_bound: float = float("inf"),
    ) -> Constraint:
        """Add a linear constraint to the model."""
        if name in self.constraints:
            raise ValueError(f"Constraint '{name}' already exists in model.")
        for v_name in coefficients.keys():
            if v_name not in self.variables:
                raise ValueError(f"Variable '{v_name}' in constraint '{name}' not declared in model.")
        
        con = Constraint(
            name=name,
            coefficients={k: float(v) for k, v in coefficients.items() if abs(v) > 1e-15},
            sense=sense,
            rhs=rhs,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            row_index=len(self._con_order),
        )
        self.constraints[name] = con
        self._con_order.append(name)
        return con

    def set_objective(
        self,
        linear_coefficients: Dict[str, float],
        sense: ObjectiveSense = ObjectiveSense.MINIMIZE,
        quadratic_coefficients: Optional[Dict[Tuple[str, str], float]] = None,
        offset: float = 0.0,
    ):
        """Define the model objective."""
        for v_name in linear_coefficients.keys():
            if v_name not in self.variables:
                raise ValueError(f"Variable '{v_name}' in objective not declared in model.")
        
        self.objective = Objective(
            sense=sense,
            linear_coefficients={k: float(v) for k, v in linear_coefficients.items() if abs(v) > 1e-15},
            quadratic_coefficients=quadratic_coefficients or {},
            offset=offset,
        )

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    @property
    def num_constraints(self) -> int:
        return len(self.constraints)

    @property
    def variable_names(self) -> List[str]:
        return list(self._var_order)

    @property
    def constraint_names(self) -> List[str]:
        return list(self._con_order)

    def validate(self) -> List[str]:
        """
        Stage 1 Model Validation:
        Checks for bound contradictions, NaN/Inf values, and empty structures.
        Returns a list of warnings or raises ModelValidationError if fatal.
        """
        warnings = []
        if self.num_variables == 0:
            raise ModelValidationError("Model has no variables.")

        for name, var in self.variables.items():
            if np.isnan(var.lower_bound) or np.isnan(var.upper_bound):
                raise ModelValidationError(f"Variable '{name}' has NaN bounds.")
            if var.lower_bound > var.upper_bound + 1e-12:
                raise ModelValidationError(
                    f"Variable '{name}' has lower bound ({var.lower_bound}) > upper bound ({var.upper_bound})."
                )
            if var.var_type == VariableType.BINARY:
                if var.lower_bound < 0 or var.upper_bound > 1:
                    warnings.append(f"Binary variable '{name}' bounds clamped to [0, 1].")

        for name, con in self.constraints.items():
            if con.lower_bound > con.upper_bound + 1e-12:
                raise ModelValidationError(
                    f"Constraint '{name}' has lower bound ({con.lower_bound}) > upper bound ({con.upper_bound})."
                )
            for v_name, coeff in con.coefficients.items():
                if np.isnan(coeff) or np.isinf(coeff):
                    raise ModelValidationError(f"Constraint '{name}' has invalid coefficient {coeff} for '{v_name}'.")

        return warnings

    def classify(self) -> str:
        """
        Stage 2 Problem Classification:
        Classifies the problem as LP, MILP, QP, or MIQP.
        """
        has_integers = any(v.is_integer for v in self.variables.values())
        has_quad = self.objective.is_quadratic

        if has_quad and has_integers:
            return "MIQP"
        elif has_quad:
            return "QP"
        elif has_integers:
            return "MILP"
        else:
            return "LP"

    def get_metadata(self) -> ModelMetadata:
        """Calculate structural statistics and features of the model."""
        p_class = self.classify()
        n_vars = self.num_variables
        n_cons = self.num_constraints
        
        nnz = sum(len(c.coefficients) for c in self.constraints.values())
        density = (nnz / (n_vars * n_cons)) if (n_vars * n_cons) > 0 else 0.0

        n_cont = sum(1 for v in self.variables.values() if v.var_type == VariableType.CONTINUOUS)
        n_int = sum(1 for v in self.variables.values() if v.var_type == VariableType.INTEGER)
        n_bin = sum(1 for v in self.variables.values() if v.var_type == VariableType.BINARY)

        all_coeffs = [abs(c) for con in self.constraints.values() for c in con.coefficients.values()]
        min_coeff = min(all_coeffs) if all_coeffs else 0.0
        max_coeff = max(all_coeffs) if all_coeffs else 0.0

        return ModelMetadata(
            name=self.name,
            problem_class=p_class,
            num_variables=n_vars,
            num_constraints=n_cons,
            num_nonzeros=nnz,
            density=density,
            num_continuous=n_cont,
            num_integer=n_int,
            num_binary=n_bin,
            num_quadratic_terms=len(self.objective.quadratic_coefficients),
            min_coefficient=min_coeff,
            max_coefficient=max_coeff,
        )

    def to_matrix_form(self) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Export constraints in canonical two-sided bound form:
        row_lb <= A * x <= row_ub
        col_lb <= x <= col_ub
        
        Returns:
            A: sp.csr_matrix of shape (num_constraints, num_variables)
            row_lb: np.ndarray of shape (num_constraints,)
            row_ub: np.ndarray of shape (num_constraints,)
            col_lb: np.ndarray of shape (num_variables,)
            col_ub: np.ndarray of shape (num_variables,)
        """
        m = self.num_constraints
        n = self.num_variables

        rows = []
        cols = []
        data = []

        row_lb = np.empty(m, dtype=np.float64)
        row_ub = np.empty(m, dtype=np.float64)

        for i, c_name in enumerate(self._con_order):
            con = self.constraints[c_name]
            row_lb[i] = con.lower_bound
            row_ub[i] = con.upper_bound
            for v_name, coeff in con.coefficients.items():
                col_idx = self.variables[v_name].column_index
                rows.append(i)
                cols.append(col_idx)
                data.append(coeff)

        A = sp.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.float64)

        col_lb = np.array([self.variables[v_name].lower_bound for v_name in self._var_order], dtype=np.float64)
        col_ub = np.array([self.variables[v_name].upper_bound for v_name in self._var_order], dtype=np.float64)

        return A, row_lb, row_ub, col_lb, col_ub

    def get_objective_vector(self) -> np.ndarray:
        """Returns linear objective vector c (normalized to MINIMIZE sense)."""
        c = np.zeros(self.num_variables, dtype=np.float64)
        multiplier = 1.0 if self.objective.sense == ObjectiveSense.MINIMIZE else -1.0
        for v_name, coeff in self.objective.linear_coefficients.items():
            c[self.variables[v_name].column_index] = coeff * multiplier
        return c

    def get_quadratic_matrix(self) -> sp.csr_matrix:
        """Returns symmetric quadratic matrix Q for 0.5 * x^T Q x."""
        n = self.num_variables
        rows, cols, data = [], [], []
        multiplier = 1.0 if self.objective.sense == ObjectiveSense.MINIMIZE else -1.0

        for (v1, v2), coeff in self.objective.quadratic_coefficients.items():
            i = self.variables[v1].column_index
            j = self.variables[v2].column_index
            scaled = coeff * multiplier
            if i == j:
                rows.append(i)
                cols.append(j)
                data.append(scaled)
            else:
                # Symmetrize off-diagonal terms
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([scaled * 0.5, scaled * 0.5])

        if not rows:
            return sp.csr_matrix((n, n), dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
