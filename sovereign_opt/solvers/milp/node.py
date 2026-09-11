"""
Branch-and-Bound Node data structures.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class Node:
    """
    Represents a subproblem in the Branch-and-Bound search tree.
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
