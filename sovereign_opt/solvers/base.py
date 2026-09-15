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
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    NODE_LIMIT = "node_limit"
    NUMERICAL_ERROR = "numerical_error"
    UNSOLVED = "unsolved"


# Statuses where a solver may still return a usable (but not proven optimal) primal point.
LIMIT_STATUSES = (SolverStatus.TIME_LIMIT, SolverStatus.ITERATION_LIMIT, SolverStatus.NODE_LIMIT)


@dataclass
class SolverResult:
    """
    Standardized result returned by all mathematical solvers.

    Dual convention (minimize-normalized): at an optimum
        grad f(x) = A^T y + z,
    where y_i > 0 means row i is active at its lower bound, y_i < 0 at its upper bound,
    and z_j (reduced cost) > 0 means variable j is at its lower bound, < 0 at its upper bound.
    For maximization models the objective is negated before these multipliers are computed.
    """
    status: SolverStatus
    objective_value: Optional[float] = None
    primal_solution: Dict[str, float] = field(default_factory=dict)
    dual_solution: Dict[str, float] = field(default_factory=dict)
    reduced_costs: Dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    runtime_seconds: float = 0.0
    nodes_explored: int = 0
    mip_gap: Optional[float] = None
    best_bound: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_optimal(self) -> bool:
        return self.status == SolverStatus.OPTIMAL

    @property
    def is_feasible(self) -> bool:
        """True only when the result carries a primal point claimed to be feasible."""
        if self.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            return True
        return self.status in LIMIT_STATUSES and bool(self.primal_solution) and bool(
            self.diagnostics.get("has_feasible_point", False)
        )


class SolverBase:
    """Abstract base class for all optimization solvers."""

    def __init__(self, name: str):
        self.name = name

    def solve(self, model, **kwargs) -> SolverResult:
        raise NotImplementedError("Subclasses must implement solve().")
