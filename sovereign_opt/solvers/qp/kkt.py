"""
Karush-Kuhn-Tucker (KKT) system builder and solver for Quadratic Programming.
"""
from typing import Tuple, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class KKTSolver:
    """
    Assembles and solves the augmented KKT linear system:
    [ Q       A_act^T ] [ x ]   = [ -c        ]
    [ A_act   0       ] [ y ]     [ b_act     ]
    """

    @classmethod
    def solve(
        cls,
        Q: np.ndarray,
        c: np.ndarray,
        A_active: np.ndarray,
        b_active: np.ndarray,
        regularization: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = Q.shape[0]
        m = A_active.shape[0]

        if m == 0:
            # Unconstrained QP: Q x = -c
            Q_reg = Q + np.eye(n) * regularization
            x = np.linalg.solve(Q_reg, -c)
            return x, np.array([])

        # Form KKT matrix
        # [ Q + reg*I    A_act^T ]
        # [ A_act        -reg*I  ]
        KKT = np.block([
            [Q + np.eye(n) * regularization, A_active.T],
            [A_active, -np.eye(m) * regularization],
        ])

        rhs = np.concatenate([-c, b_active])

        sol = np.linalg.solve(KKT, rhs)
        x = sol[:n]
        y = sol[n:]
        return x, y
