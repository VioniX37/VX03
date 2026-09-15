"""
Branch-and-Bound Node data structures.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class Node:
    """
    Represents a subproblem in the Branch-and-Bound search tree (name-based, kept for compatibility).
    """
    node_id: int
    parent_id: Optional[int]
    depth: int
    bounds: Dict[str, Tuple[float, float]]  # var_name -> (lb, ub)
    lower_bound: float = float("-inf")
    solution: Dict[str, float] = field(default_factory=dict)
    is_integer_feasible: bool = False
    is_pruned: bool = False

    def __lt__(self, other: "Node") -> bool:
        # For PriorityQueue min-heap (best lower bound first)
        return self.lower_bound < other.lower_bound


@dataclass(eq=False)
class TreeNode:
    """
    Index-based branch-and-cut node (v2).

    Bounds are stored as a chain of changes relative to the parent, so memory per node is
    O(changes) instead of O(n). The parent's optimal basis is kept for a warm-started
    dual simplex solve and released once the node has been processed.
    """
    node_id: int
    parent: Optional["TreeNode"]
    depth: int
    changes: Tuple[Tuple[int, float, float], ...] = ()  # (column, lower, upper) applied on top of parent
    bound: float = float("-inf")  # internal (minimize-normalized) lower bound estimate
    basis: Optional[Tuple[np.ndarray, np.ndarray]] = None
    parent_objective: float = float("nan")
    parent_fraction: float = 0.0
    branch_column: int = -1
    direction: int = 0  # -1 down branch, +1 up branch
    label: str = "Root"
    branch_name: str = "Root"

    def __lt__(self, other: "TreeNode") -> bool:
        return (self.bound, -self.depth) < (other.bound, -other.depth)
