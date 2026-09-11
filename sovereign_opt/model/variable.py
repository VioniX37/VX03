"""
Variable representation for optimization models.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


@dataclass
class Variable:
    """
    Represents an optimization decision variable.
    
    Attributes:
        name: Unique identifier for the variable.
        lower_bound: Lower bound (default 0.0).
        upper_bound: Upper bound (default +inf).
        var_type: CONTINUOUS, INTEGER, or BINARY.
        column_index: Index in the coefficient matrix (assigned by model).
    """
    name: str
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    var_type: VariableType = VariableType.CONTINUOUS
    column_index: int = -1

    def __post_init__(self):
        if self.var_type == VariableType.BINARY:
            # Binary variables are constrained to [0, 1] integers
            self.lower_bound = max(0.0, self.lower_bound)
            self.upper_bound = min(1.0, self.upper_bound)

    @property
    def is_integer(self) -> bool:
        return self.var_type in (VariableType.INTEGER, VariableType.BINARY)

    @property
    def is_binary(self) -> bool:
        return self.var_type == VariableType.BINARY

    @property
    def is_fixed(self) -> bool:
        return abs(self.lower_bound - self.upper_bound) < 1e-12
