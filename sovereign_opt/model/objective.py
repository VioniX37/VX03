"""
Objective representation for optimization models.
Supports linear and convex quadratic objectives.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Objective:
    """
    Represents an optimization objective:
    sense * ( 0.5 * x^T Q x + c^T x + offset )
    
    Attributes:
        sense: MINIMIZE or MAXIMIZE.
        linear_coefficients: Mapping of var_name -> linear coefficient c_j.
        quadratic_coefficients: Mapping of (var_i, var_j) -> quadratic coefficient Q_ij.
        offset: Constant scalar offset.
    """
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    linear_coefficients: Dict[str, float] = field(default_factory=dict)
    quadratic_coefficients: Dict[Tuple[str, str], float] = field(default_factory=dict)
    offset: float = 0.0

    @property
    def is_quadratic(self) -> bool:
        return len(self.quadratic_coefficients) > 0

    @property
    def is_linear(self) -> bool:
        return len(self.quadratic_coefficients) == 0

    def add_linear_term(self, var_name: str, coefficient: float):
        self.linear_coefficients[var_name] = self.linear_coefficients.get(var_name, 0.0) + coefficient

    def add_quadratic_term(self, var1: str, var2: str, coefficient: float):
        # Store in sorted tuple key for symmetry
        key = (var1, var2) if var1 <= var2 else (var2, var1)
        self.quadratic_coefficients[key] = self.quadratic_coefficients.get(key, 0.0) + coefficient
