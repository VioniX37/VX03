"""
Constraint representation for optimization models.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class ConstraintSense(str, Enum):
    LE = "<="
    GE = ">="
    EQ = "=="
    RANGE = "range"


@dataclass
class Constraint:
    """
    Represents a linear constraint in the form:
    lower_bound <= sum(coefficients[var] * var) <= upper_bound
    """
    name: str
    coefficients: Dict[str, float] = field(default_factory=dict)
    sense: ConstraintSense = ConstraintSense.LE
    rhs: float = 0.0
    lower_bound: float = float("-inf")
    upper_bound: float = float("inf")
    row_index: int = -1

    def __post_init__(self):
        # Only populate bounds from rhs if they were not explicitly specified
        if self.sense == ConstraintSense.LE:
            if self.upper_bound == float("inf"):
                self.upper_bound = self.rhs
        elif self.sense == ConstraintSense.GE:
            if self.lower_bound == float("-inf"):
                self.lower_bound = self.rhs
        elif self.sense == ConstraintSense.EQ:
            if self.lower_bound == float("-inf") and self.upper_bound == float("inf"):
                self.lower_bound = self.rhs
                self.upper_bound = self.rhs
        elif self.sense == ConstraintSense.RANGE:
            pass

    @property
    def is_equality(self) -> bool:
        return self.sense == ConstraintSense.EQ or (
            self.lower_bound == self.upper_bound and not (
                self.lower_bound == float("-inf") or self.upper_bound == float("inf")
            )
        )
