"""
Base interfaces and result classes for sovereign solvers.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any, List
import numpy as np


class SolverStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    NUMERICAL_ERROR = "numerical_error"
    UNSOLVED = "unsolved"


@dataclass
class SolverResult:
    """
    Standardized result returned by all mathematical solvers.
    """
    status: SolverStatus
    objective_value: Optional[float] = None
    primal_solution: Dict[str, float] = field(default_factory=dict)
    dual_solution: Dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    runtime_seconds: float = 0.0
    nodes_explored: int = 0
    mip_gap: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_optimal(self) -> bool:
        return self.status == SolverStatus.OPTIMAL

    @property
    def is_feasible(self) -> bool:
        return self.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)


class SolverBase:
    """Abstract base class for all optimization solvers."""

    def __init__(self, name: str):
        self.name = name

    def solve(self, model, **kwargs) -> SolverResult:
        raise NotImplementedError("Subclasses must implement solve().")
