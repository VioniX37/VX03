"""
MILP scaling benchmark: industrial-style MILPs from hundreds to tens of thousands of variables,
sovereign branch-and-cut vs HiGHS (single thread), same time limit.

    python -m benchmarks.milp_scale
    python -m benchmarks.milp_scale --time-limit 300
"""
import argparse
import json
import os
import sys
import time

from benchmarks.compare import RESULTS, _run, machine_info

CASES = [
    ("unit_commitment_G10_T24", "uc", (10, 24)),
    ("unit_commitment_G20_T48", "uc", (20, 48)),
    ("unit_commitment_G40_T72", "uc", (40, 72)),
    ("unit_commitment_G60_T168", "uc", (60, 168)),
    ("facility_location_W20_C100", "fl", (20, 100)),
    ("facility_location_W40_C300", "fl", (40, 300)),
    ("facility_location_W80_C800", "fl", (80, 800)),
]


def build(kind, args):
    if kind == "uc":
        from benchmarks.industrial.power_dispatch import build_unit_commitment_fleet
        return build_unit_commitment_fleet(*args)
    from benchmarks.industrial.supply_chain import build_facility_location_model
    return build_facility_location_model(*args)


def _cache_path(name):
    return os.path.join(os.path.dirname(__file__), "data", "cache", f"{name}.mps")


def _ours(name, kind, args, time_limit, q):
    try:
        from sovereign_opt.solvers.dispatch import solve_model
        m = build(kind, args)
        t0 = time.perf_counter()
        out = solve_model(m, time_limit_seconds=time_limit)
        r = out.result
        q.put({"status": r.status.value, "objective": r.objective_value, "time": time.perf_counter() - t0,
               "best_bound": r.best_bound, "mip_gap": r.mip_gap, "nodes": r.nodes_explored,
               "certificate": out.certificate.status})
    except Exception as exc:
        q.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def _highs(name, kind, args, time_limit, q):
    try:
        import highspy
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", float(time_limit))
        h.readModel(_cache_path(name))
        t0 = time.perf_counter()
        h.run()
        info = h.getInfo()
        st = h.modelStatusToString(h.getModelStatus()).lower()
        q.put({"status": st, "objective": info.objective_function_value if info.primal_solution_status >= 1 else None,
               "time": time.perf_counter() - t0, "mip_gap": info.mip_gap, "nodes": int(info.mip_node_count)})
    except Exception as exc:
        q.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    from sovereign_opt.parsers.mps_writer import write_mps
    rows = []
    for name, kind, a in CASES:
        if args.only and name not in args.only:
            continue
        m = build(kind, a)
        os.makedirs(os.path.dirname(_cache_path(name)), exist_ok=True)
        with open(_cache_path(name), "w") as f:
            f.write(write_mps(m))
        n_int = sum(1 for v in m.variables.values() if v.is_integer)
        hard = args.time_limit * 1.5 + 60
        highs = _run(_highs, (name, kind, a, args.time_limit), hard)
        ours = _run(_ours, (name, kind, a, args.time_limit), hard)
        ref = highs.get("objective")
        err = abs(ours["objective"] - ref) / max(1.0, abs(ref)) if (ref is not None and ours.get("objective") is not None) else None
        rows.append({"name": name, "cols": m.num_variables, "rows": m.num_constraints, "int_vars": n_int,
                     "ours": ours, "highs": highs, "rel_error_vs_highs": err})
        print(f"{name:30s} {m.num_variables:>7d} vars ({n_int:>6d} int) | ours {ours.get('status')!s:10s} "
              f"{ours.get('time', 0):8.2f}s gap {ours.get('mip_gap')} | HiGHS {highs.get('status')!s:18s} "
              f"{highs.get('time', 0):8.2f}s gap {highs.get('mip_gap')} | obj err {err}", flush=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"milpscale_{stamp}.json")
    with open(path, "w") as f:
        json.dump({"meta": {"timestamp": stamp, "machine": machine_info(), "time_limit": args.time_limit}, "rows": rows},
                  f, indent=1, default=str)
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
