"""
Scaling benchmark: how large an LP can the engine solve, and how fast, on this machine?

    python -m benchmarks.scale                          # 1k .. 1M variables, CPU
    python -m benchmarks.scale --sizes 1e5 1e6 3e6      # on a GPU machine (Kaggle): adds PDLP-CUDA

Model: the multi-product production-distribution network in benchmarks/industrial/supply_chain.py.
Methods (each in its own process with a hard time limit):
    highs            HiGHS default LP solver, single thread (comparison reference)
    dual_simplex     sovereign dual simplex (full pipeline incl. certificate), small sizes only
    interior_point   sovereign HSD interior point + crossover, small / medium sizes
    pdlp_cpu         sovereign PDLP on CPU (NumPy / SciPy sparse)
    pdlp_cuda        sovereign PDLP on GPU (PyTorch CUDA sparse), when a GPU is present
PDLP runs to relative KKT tolerance --pdlp-tol (1e-4 is the usual large-scale target; the objective
error vs HiGHS is reported so the accuracy is visible).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict

import numpy as np

from benchmarks.compare import RESULTS, machine_info

MAX_VARS = {"dual_simplex": 25_000, "interior_point": 60_000}


def _highs_arrays(lp, time_limit):
    import highspy
    A = lp.A.tocsc()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(time_limit))
    inf = highspy.kHighsInf
    L = highspy.HighsLp()
    L.num_col_, L.num_row_ = A.shape[1], A.shape[0]
    L.col_cost_ = lp.c
    L.col_lower_ = np.where(np.isinf(lp.col_lb), -inf, lp.col_lb)
    L.col_upper_ = np.where(np.isinf(lp.col_ub), inf, lp.col_ub)
    L.row_lower_ = np.where(np.isinf(lp.row_lb), -inf, lp.row_lb)
    L.row_upper_ = np.where(np.isinf(lp.row_ub), inf, lp.row_ub)
    L.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    L.a_matrix_.start_, L.a_matrix_.index_, L.a_matrix_.value_ = A.indptr, A.indices, A.data
    L.a_matrix_.num_col_, L.a_matrix_.num_row_ = A.shape[1], A.shape[0]
    h.passModel(L)
    t0 = time.perf_counter()
    h.run()
    t = time.perf_counter() - t0
    st = h.modelStatusToString(h.getModelStatus()).lower()
    return {"status": st, "objective": h.getInfo().objective_function_value if st == "optimal" else None, "time": t}


def _worker(method: str, target: int, time_limit: float, pdlp_tol: float, q):
    try:
        from benchmarks.industrial.supply_chain import build_supply_chain_arrays, sizes_for, arrays_to_model
        lp = build_supply_chain_arrays(*sizes_for(target))
        if method == "highs":
            q.put(_highs_arrays(lp, time_limit))
            return
        if method.startswith("pdlp"):
            from sovereign_opt.solvers.lp.pdlp import pdlp
            device = "cuda" if method == "pdlp_cuda" else "cpu"
            if device == "cuda":  # warm up the CUDA context so it is not billed to the solve
                import torch
                torch.zeros(1, device="cuda")
            r = pdlp(lp.A, lp.c, lp.row_lb, lp.row_ub, lp.col_lb, lp.col_ub, tol=pdlp_tol,
                     time_limit=time_limit, device=device)
            q.put({"status": r.status, "objective": r.primal_objective, "time": r.runtime, "setup_time": r.setup_time,
                   "iterations": r.iterations, "restarts": r.restarts, "device": r.device,
                   "rel_primal_residual": r.rel_primal_residual, "rel_dual_residual": r.rel_dual_residual,
                   "rel_gap": r.rel_gap})
            return
        from sovereign_opt.solvers.dispatch import solve_model
        model = arrays_to_model(lp)
        t0 = time.perf_counter()
        out = solve_model(model, algorithm=method, time_limit_seconds=time_limit)
        q.put({"status": out.result.status.value, "objective": out.result.objective_value,
               "time": time.perf_counter() - t0, "iterations": out.result.iterations,
               "certificate": out.certificate.status})
    except Exception as exc:
        q.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def run(method, target, time_limit, pdlp_tol) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(method, target, time_limit, pdlp_tol, q))
    t0 = time.perf_counter()
    p.start()
    res = None
    hard = time_limit * 1.5 + 120
    while time.perf_counter() - t0 < hard:
        try:
            res = q.get(timeout=1.0)
            break
        except Exception:
            if not p.is_alive():
                break
    if res is None:
        p.terminate()
        res = {"status": "killed", "time": time.perf_counter() - t0}
    p.join(5)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="*", type=float, default=[1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--pdlp-tol", type=float, default=1e-4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    info = machine_info()
    methods = args.methods or (["highs", "dual_simplex", "interior_point", "pdlp_cpu"]
                               + (["pdlp_cuda"] if info.get("cuda") else []))
    from benchmarks.industrial.supply_chain import build_supply_chain_arrays, sizes_for
    rows = []
    for s in args.sizes:
        target = int(s)
        lp = build_supply_chain_arrays(*sizes_for(target))
        m, n = lp.A.shape
        row = {"target": target, "rows": m, "cols": n, "nnz": int(lp.A.nnz), "results": {}}
        for method in methods:
            if n > MAX_VARS.get(method, 10**12):
                row["results"][method] = {"status": "skipped (size)"}
                continue
            r = run(method, target, args.time_limit, args.pdlp_tol)
            row["results"][method] = r
        ref = row["results"].get("highs", {}).get("objective")
        for r in row["results"].values():
            if ref is not None and r.get("objective") is not None:
                r["rel_error_vs_highs"] = abs(r["objective"] - ref) / max(1.0, abs(ref))
        rows.append(row)
        print(f"{n:>9d} vars {m:>8d} rows {lp.A.nnz:>9d} nnz | " + " | ".join(
            f"{k} {v.get('status')} {v.get('time', float('nan')):.2f}s" for k, v in row["results"].items()), flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(RESULTS, f"scale{('_' + args.tag) if args.tag else ''}_{stamp}.json")
    with open(path, "w") as f:
        json.dump({"meta": {"timestamp": stamp, "machine": info, "time_limit": args.time_limit,
                            "pdlp_tol": args.pdlp_tol, "model": "supply_chain production-distribution LP"},
                   "rows": rows}, f, indent=1, default=str)
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
