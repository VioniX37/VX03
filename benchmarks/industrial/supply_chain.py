"""
Scalable industrial benchmark: multi-product production-distribution network (LP).

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
of variables); `build_supply_chain_model` wraps small instances as an OptimizationModel.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense


@dataclass
class LPArrays:
    name: str
    A: sp.csr_matrix
    c: np.ndarray
    row_lb: np.ndarray
    row_ub: np.ndarray
    col_lb: np.ndarray
    col_ub: np.ndarray
    meta: dict = field(default_factory=dict)  # generator data (sizes, nearest warehouses, demand, capacities)

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
                    col_lb=np.zeros(n), col_ub=np.full(n, np.inf),
                    meta={"plants": P, "warehouses": W, "customers": C, "products": K, "neighbours": L,
                          "near": near, "distance": dist, "demand": demand, "plant_capacity": cap_p,
                          "warehouse_throughput": thr_w})


def arrays_to_model(lp: LPArrays) -> OptimizationModel:
    """Wrap arrays as an OptimizationModel (intended for up to ~100k variables)."""
    m, n = lp.A.shape
    model = OptimizationModel(name=lp.name)
    names = [f"v{j}" for j in range(n)]
    for j in range(n):
        model.add_variable(names[j], float(lp.col_lb[j]), float(lp.col_ub[j]))
    A = lp.A.tocsr()
    for i in range(m):
        s, e = A.indptr[i], A.indptr[i + 1]
        coeffs = {names[j]: float(v) for j, v in zip(A.indices[s:e], A.data[s:e])}
        lo, hi = lp.row_lb[i], lp.row_ub[i]
        if lo == hi:
            model.add_constraint(f"r{i}", coeffs, ConstraintSense.EQ, rhs=float(lo))
        elif np.isinf(lo):
            model.add_constraint(f"r{i}", coeffs, ConstraintSense.LE, rhs=float(hi))
        elif np.isinf(hi):
            model.add_constraint(f"r{i}", coeffs, ConstraintSense.GE, rhs=float(lo))
        else:
            model.add_constraint(f"r{i}", coeffs, ConstraintSense.RANGE, lower_bound=float(lo), upper_bound=float(hi))
    model.set_objective({names[j]: float(v) for j, v in enumerate(lp.c) if v != 0.0}, ObjectiveSense.MINIMIZE)
    return model


def build_supply_chain_model(target_vars: int = 2000, products: int = 4, neighbours: int = 5) -> OptimizationModel:
    P, W, C = sizes_for(target_vars, products, neighbours)
    return arrays_to_model(build_supply_chain_arrays(P, W, C, products, neighbours))


def build_facility_location_model(warehouses: int = 30, customers: int = 150, products: int = 2,
                                  neighbours: int = 6, seed: int = 11) -> OptimizationModel:
    """
    Capacitated warehouse location (MILP): decide which warehouses to open (binary y_w, fixed cost) and
    route each customer's demand from nearby open warehouses. Big-M linking z[w,c,k] <= d[c,k] * y_w makes
    the LP relaxation weak, which is what makes this class hard for branch-and-bound.
    """
    from sovereign_opt.model.variable import VariableType
    rng = np.random.default_rng(seed)
    W, C, K, L = warehouses, customers, products, min(neighbours, warehouses)
    loc_w = rng.uniform(0, 100, (W, 2))
    loc_c = rng.uniform(0, 100, (C, 2))
    demand = rng.uniform(5, 40, (C, K))
    cap = rng.uniform(0.8, 1.4, W) * demand.sum() / W * 3.0
    fixed = rng.uniform(800, 1600, W) * (cap / cap.mean())
    dist = np.linalg.norm(loc_c[:, None, :] - loc_w[None, :, :], axis=2)
    near = np.argsort(dist, axis=1)[:, :L]

    model = OptimizationModel(name=f"Facility_Location_W{W}_C{C}_K{K}")
    obj = {}
    for w in range(W):
        model.add_variable(f"y_{w}", 0.0, 1.0, VariableType.BINARY)
        obj[f"y_{w}"] = float(fixed[w])
    through = {w: {} for w in range(W)}
    for c in range(C):
        for k in range(K):
            dem = {}
            for w in near[c]:
                z = f"z_{w}_{c}_{k}"
                model.add_variable(z, 0.0, float(demand[c, k]))
                obj[z] = float(dist[c, w] * 0.05)
                dem[z] = 1.0
                through[w][z] = 1.0
                model.add_constraint(f"link_{w}_{c}_{k}", {z: 1.0, f"y_{w}": -float(demand[c, k])},
                                     ConstraintSense.LE, rhs=0.0)
            model.add_constraint(f"demand_{c}_{k}", dem, ConstraintSense.GE, rhs=float(demand[c, k]))
    for w in range(W):
        coeffs = dict(through[w])
        coeffs[f"y_{w}"] = -float(cap[w])
        model.add_constraint(f"cap_{w}", coeffs, ConstraintSense.LE, rhs=0.0)
    model.set_objective(obj, ObjectiveSense.MINIMIZE)
    return model
