"""
Scalable industrial benchmark: multi-product production-distribution network (LP) using Gurobi.

Plants make products, ship them to regional warehouses, and warehouses serve customers.
Each customer can be served by its `neighbours` nearest warehouses. Structure follows the classic
production-distribution models in the supply-chain literature; the data is synthetic (fixed seed),
so any size can be generated reproducibly.

Decision variables (all continuous, >= 0):
    x[p, w, k]  units of product k shipped plant p -> warehouse w
    z[w, c, k]  units of product k shipped warehouse w -> customer c (only for nearby w)
Constraints:
    plant capacity       sum_{w,k} x[p,w,k]            <= cap_p
    warehouse balance    sum_p x[p,w,k] - sum_c z[w,c,k] = 0        for every (w, k)
    warehouse throughput sum_{c,k} z[w,c,k]            <= thr_w
    customer demand      sum_w z[w,c,k]                >= d[c,k]
Objective: minimise production-weighted transport cost (distance based).

`build_supply_chain_arrays` builds the sparse matrix directly (vectorised, suitable for millions
of variables); `arrays_to_model` builds a Gurobi Optimization Model from arrays.
"""
from dataclasses import dataclass
import numpy as np

# Compatibility workaround for scipy/numpy version mismatch in environment
if not hasattr(np, "long"):
    setattr(np, "long", np.int64)
if not hasattr(np, "ulong"):
    setattr(np, "ulong", np.uint64)

try:
    np.array([1], copy=None)
except ValueError:
    _orig_array = np.array
    def _compat_array(object, dtype=None, copy=None, order='K', subok=False, ndmin=0, **kwargs):
        if copy is None:
            copy = False
        return _orig_array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, **kwargs)
    np.array = _compat_array

import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB


@dataclass
class LPArrays:
    name: str
    A: sp.csr_matrix
    c: np.ndarray
    row_lb: np.ndarray
    row_ub: np.ndarray
    col_lb: np.ndarray
    col_ub: np.ndarray

    @property
    def shape(self):
        return self.A.shape


def sizes_for(target_vars: int, products: int = 4, neighbours: int = 5):
    """Pick (plants, warehouses, customers) so the model has about `target_vars` variables."""
    customers = max(10, int(target_vars / (products * (neighbours + 0.2))))
    warehouses = max(neighbours, int(np.sqrt(customers)))
    plants = max(2, warehouses // 5)
    return plants, warehouses, customers


def build_supply_chain_arrays(plants: int, warehouses: int, customers: int, products: int = 4,
                              neighbours: int = 5, seed: int = 7) -> LPArrays:
    rng = np.random.default_rng(seed)
    P, W, C, K, L = plants, warehouses, customers, products, min(neighbours, warehouses)
    loc_p = rng.uniform(0, 100, (P, 2))
    loc_w = rng.uniform(0, 100, (W, 2))
    loc_c = rng.uniform(0, 100, (C, 2))

    demand = rng.uniform(5, 50, (C, K))
    total = demand.sum()
    cap_p = rng.uniform(0.8, 1.2, P)
    cap_p = cap_p / cap_p.sum() * total * 1.25
    thr_w = rng.uniform(0.8, 1.2, W)
    thr_w = thr_w / thr_w.sum() * total * 1.6

    d_pw = np.linalg.norm(loc_p[:, None, :] - loc_w[None, :, :], axis=2)  # P x W
    # nearest L warehouses per customer, chunked to keep memory flat
    near = np.empty((C, L), dtype=np.int64)
    dist = np.empty((C, L))
    for s in range(0, C, 20000):
        blk = np.linalg.norm(loc_c[s:s + 20000, None, :] - loc_w[None, :, :], axis=2)
        idx = np.argpartition(blk, L - 1, axis=1)[:, :L]
        near[s:s + 20000] = idx
        dist[s:s + 20000] = np.take_along_axis(blk, idx, axis=1)
    prod_cost = rng.uniform(0.5, 1.5, K)

    n_x = P * W * K
    n_z = C * L * K
    n = n_x + n_z
    # column index helpers
    xi = np.arange(n_x).reshape(P, W, K)
    zi = (n_x + np.arange(n_z)).reshape(C, L, K)

    c = np.concatenate([(d_pw[:, :, None] * 0.8 + prod_cost[None, None, :]).ravel(),
                        np.repeat(dist[:, :, None], K, axis=2).ravel()])

    rows, cols, vals, rlb, rub = [], [], [], [], []
    r0 = 0
    # plant capacity: P rows
    rows.append(np.repeat(np.arange(P), W * K) + r0)
    cols.append(xi.reshape(P, -1).ravel())
    vals.append(np.ones(n_x))
    rlb.append(np.full(P, -np.inf))
    rub.append(cap_p)
    r0 += P
    # warehouse balance: W*K rows, row id = r0 + w*K + k
    bal = r0 + np.arange(W * K).reshape(W, K)
    rows.append(np.broadcast_to(bal[None, :, :], (P, W, K)).ravel())
    cols.append(xi.ravel())
    vals.append(np.ones(n_x))
    rows.append(bal[near[:, :, None], np.arange(K)[None, None, :]].ravel())
    cols.append(zi.ravel())
    vals.append(-np.ones(n_z))
    rlb.append(np.zeros(W * K))
    rub.append(np.zeros(W * K))
    r0 += W * K
    # warehouse throughput: W rows
    rows.append(np.repeat(near[:, :, None] + r0, K, axis=2).ravel())
    cols.append(zi.ravel())
    vals.append(np.ones(n_z))
    rlb.append(np.full(W, -np.inf))
    rub.append(thr_w)
    r0 += W
    # customer demand: C*K rows
    dem = r0 + np.arange(C * K).reshape(C, K)
    rows.append(np.broadcast_to(dem[:, None, :], (C, L, K)).ravel())
    cols.append(zi.ravel())
    vals.append(np.ones(n_z))
    rlb.append(demand.ravel())
    rub.append(np.full(C * K, np.inf))
    r0 += C * K

    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, n))
    return LPArrays(name=f"supply_chain_P{P}_W{W}_C{C}_K{K}", A=A, c=c,
                    row_lb=np.concatenate(rlb), row_ub=np.concatenate(rub),
                    col_lb=np.zeros(n), col_ub=np.full(n, np.inf))


def arrays_to_model(lp: LPArrays) -> gp.Model:
    """Wrap arrays as a Gurobi Model (intended for up to millions of variables)."""
    m, n = lp.A.shape
    model = gp.Model(lp.name)
    names = [f"v{j}" for j in range(n)]

    vars_list = model.addVars(n, lb=lp.col_lb, ub=lp.col_ub, obj=lp.c, vtype=GRB.CONTINUOUS, name=names)
    model.update()

    A = lp.A.tocsr()
    for i in range(m):
        s, e = A.indptr[i], A.indptr[i + 1]
        expr = gp.LinExpr([(float(v), vars_list[j]) for j, v in zip(A.indices[s:e], A.data[s:e])])
        lo, hi = lp.row_lb[i], lp.row_ub[i]
        if lo == hi:
            model.addConstr(expr == float(lo), name=f"r{i}")
        elif np.isinf(lo):
            model.addConstr(expr <= float(hi), name=f"r{i}")
        elif np.isinf(hi):
            model.addConstr(expr >= float(lo), name=f"r{i}")
        else:
            model.addRange(expr, float(lo), float(hi), name=f"r{i}")

    model.ModelSense = GRB.MINIMIZE
    return model


def build_supply_chain_model(target_vars: int = 2000, products: int = 4, neighbours: int = 5) -> gp.Model:
    P, W, C = sizes_for(target_vars, products, neighbours)
    return arrays_to_model(build_supply_chain_arrays(P, W, C, products, neighbours))


def build_facility_location_model(warehouses: int = 30, customers: int = 150, products: int = 2,
                                  neighbours: int = 6, seed: int = 11) -> gp.Model:
    """
    Capacitated warehouse location (MILP) using Gurobi.
    """
    rng = np.random.default_rng(seed)
    W, C, K, L = warehouses, customers, products, min(neighbours, warehouses)
    loc_w = rng.uniform(0, 100, (W, 2))
    loc_c = rng.uniform(0, 100, (C, 2))
    demand = rng.uniform(5, 40, (C, K))
    cap = rng.uniform(0.8, 1.4, W) * demand.sum() / W * 3.0
    fixed = rng.uniform(800, 1600, W) * (cap / cap.mean())
    dist = np.linalg.norm(loc_c[:, None, :] - loc_w[None, :, :], axis=2)
    near = np.argsort(dist, axis=1)[:, :L]

    model = gp.Model(f"Facility_Location_W{W}_C{C}_K{K}")
    obj_expr = gp.LinExpr()

    y_vars = {}
    for w in range(W):
        y = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"y_{w}")
        y_vars[w] = y
        obj_expr += float(fixed[w]) * y

    through = {w: [] for w in range(W)}

    for c in range(C):
        for k in range(K):
            dem_vars = []
            for w in near[c]:
                z = model.addVar(lb=0.0, ub=float(demand[c, k]), vtype=GRB.CONTINUOUS, name=f"z_{w}_{c}_{k}")
                obj_expr += float(dist[c, w] * 0.05) * z
                dem_vars.append(z)
                through[w].append(z)
                # Link constraint: z - demand[c, k] * y <= 0
                model.addConstr(z - float(demand[c, k]) * y_vars[w] <= 0.0, name=f"link_{w}_{c}_{k}")

            model.addConstr(gp.quicksum(dem_vars) >= float(demand[c, k]), name=f"demand_{c}_{k}")

    for w in range(W):
        model.addConstr(
            gp.quicksum(through[w]) - float(cap[w]) * y_vars[w] <= 0.0,
            name=f"cap_{w}"
        )

    model.setObjective(obj_expr, GRB.MINIMIZE)
    return model


import time


if __name__ == "__main__":
    print("Building Supply Chain LP model (Gurobi)...")
    t0 = time.perf_counter()
    m_sc = build_supply_chain_model(2000)
    m_sc.optimize()
    elapsed = time.perf_counter() - t0
    if m_sc.Status == GRB.OPTIMAL:
        print(f"Supply Chain LP Optimal Objective: {m_sc.ObjVal:.6f}")
        print(f"Supply Chain LP Gurobi Runtime:   {m_sc.Runtime:.6f} seconds")
        print(f"Supply Chain LP Wall-Clock Time:   {elapsed:.6f} seconds")

    print("\nBuilding Facility Location MILP model (Gurobi)...")
    t0 = time.perf_counter()
    m_fl = build_facility_location_model(warehouses=20, customers=100)
    m_fl.optimize()
    elapsed = time.perf_counter() - t0
    if m_fl.Status == GRB.OPTIMAL:
        print(f"Facility Location MILP Optimal Objective: {m_fl.ObjVal:.6f}")
        print(f"Facility Location MILP Gurobi Runtime:   {m_fl.Runtime:.6f} seconds")
        print(f"Facility Location MILP Wall-Clock Time:   {elapsed:.6f} seconds")

