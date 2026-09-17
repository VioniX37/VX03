"""
Side-by-side benchmark: Sovereign engine vs HiGHS on public benchmark instances.

    python -m benchmarks.compare --suite netlib
    python -m benchmarks.compare --suite miplib --time-limit 120
    python -m benchmarks.compare --suite quick --jobs 4

HiGHS (the leading open-source LP/MIP/QP solver) is used here ONLY as the comparison
reference. It is never imported by the sovereign_opt package; the guard test in
tests/test_benchmarks.py enforces that.

Every solve runs in its own process with a hard wall-clock limit, so a stuck run cannot take
the harness down. Parse time is reported separately and excluded from solve time for both
solvers. Results go to benchmarks/results/<suite>_<timestamp>.json and a Markdown table.
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from benchmarks.instances import NOTES, SUITES, Instance, get_instance, mps_path

RESULTS = os.path.join(os.path.dirname(__file__), "results")
LP_TOL = 1e-6
MIP_TOL = 1e-4


def machine_info() -> Dict[str, Any]:
    info = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0]}
    try:
        from sovereign_opt.runtime.device import DeviceDetector
        d = DeviceDetector.get_info()
        info.update({"cuda": d.has_cuda, "gpu": d.cuda_device_name})
    except Exception:
        pass
    return info


# ------------------------------------------------------------------------------ workers
def _ours(key: str, algorithm: str, time_limit: float, options: Dict[str, Any], q):
    try:
        from sovereign_opt.parsers.mps_parser import MPSParser
        from sovereign_opt.solvers.dispatch import solve_model
        inst = get_instance(key)
        t0 = time.perf_counter()
        model = MPSParser.parse_file(mps_path(inst))
        parse = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = solve_model(model, algorithm=algorithm, time_limit_seconds=time_limit, **options)
        total = time.perf_counter() - t0
        r, c = out.result, out.certificate
        q.put({
            "status": r.status.value, "objective": r.objective_value, "time": total, "parse_time": parse,
            "algorithm": out.algorithm, "iterations": r.iterations, "nodes": r.nodes_explored,
            "mip_gap": r.mip_gap, "best_bound": r.best_bound, "certificate": c.status,
            "max_primal_violation": c.max_primal_violation, "duality_gap": c.duality_gap,
            "timings": out.timings, "class": out.problem_class,
            "diagnostics": {k: v for k, v in r.diagnostics.items() if isinstance(v, (int, float, str, bool))},
        })
    except Exception as exc:  # reported, never hidden
        q.put({"status": "error", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]})


def _highs(key: str, time_limit: float, q):
    try:
        import highspy
        inst = get_instance(key)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("threads", 1)
        t0 = time.perf_counter()
        h.readModel(mps_path(inst))
        parse = time.perf_counter() - t0
        t0 = time.perf_counter()
        h.run()
        total = time.perf_counter() - t0
        info = h.getInfo()
        status = h.modelStatusToString(h.getModelStatus()).lower()
        obj = info.objective_function_value if info.primal_solution_status >= 1 else None
        q.put({"status": status, "objective": obj, "time": total, "parse_time": parse,
               "iterations": int(max(info.simplex_iteration_count, 0) + max(info.ipm_iteration_count, 0)),
               "nodes": int(max(info.mip_node_count, 0)), "mip_gap": info.mip_gap if info.mip_node_count >= 0 else None})
    except Exception as exc:
        q.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def _run(target, args, hard_limit: float) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=args + (q,))
    t0 = time.perf_counter()
    p.start()
    res: Optional[Dict[str, Any]] = None
    while time.perf_counter() - t0 < hard_limit:
        try:
            res = q.get(timeout=0.5)
            break
        except Exception:
            if not p.is_alive():
                break
    if res is None:
        p.terminate()
        res = {"status": "killed" if p.is_alive() or p.exitcode is None else "crashed",
               "time": time.perf_counter() - t0, "error": f"no result within {hard_limit:.0f}s (exit={p.exitcode})"}
    p.join(5)
    return res


# ------------------------------------------------------------------------------ verdict
def _rel(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(a - b) / max(1.0, abs(b))


def verdict(inst: Instance, ours: Dict[str, Any], highs: Optional[Dict[str, Any]], is_mip: bool) -> Dict[str, Any]:
    tol = MIP_TOL if is_mip else LP_TOL
    ref = inst.published_optimum
    ref_src = "published"
    if highs and highs.get("status") == "optimal" and highs.get("objective") is not None:
        if ref is None:
            ref, ref_src = highs["objective"], "HiGHS"
    out = {"reference": ref, "reference_source": ref_src if ref is not None else None,
           "rel_error": _rel(ours.get("objective"), ref),
           "rel_error_vs_highs": _rel(ours.get("objective"), (highs or {}).get("objective"))}
    st = ours.get("status")
    if inst.expected_status == "infeasible":
        out["verdict"] = "correct (infeasible detected)" if st in ("infeasible", "infeasible_or_unbounded") else f"wrong ({st})"
        out["ok"] = out["verdict"].startswith("correct")
        return out
    errs = [e for e in (out["rel_error"], out["rel_error_vs_highs"]) if e is not None]
    close = bool(errs) and min(errs) <= tol
    if st == "optimal" and close and ours.get("certificate") == "OPTIMAL_CERTIFIED":
        out["verdict"], out["ok"] = "match (certified)", True
    elif st == "optimal" and close:
        out["verdict"], out["ok"] = f"match ({ours.get('certificate')})", True
    elif st in ("time_limit", "node_limit", "feasible") and ours.get("objective") is not None:
        out["verdict"], out["ok"] = f"limit, incumbent err {min(errs) if errs else float('nan'):.1e}", False
    elif st in ("killed", "crashed", "error"):
        out["verdict"], out["ok"] = st, False
    else:
        out["verdict"], out["ok"] = f"mismatch ({st})", False
    return out


# ------------------------------------------------------------------------------ report
def _fmt_t(t: Optional[float]) -> str:
    return "-" if t is None else (f"{t * 1000:.0f} ms" if t < 1 else f"{t:.2f} s")


def _fmt_o(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.10g}"


def markdown(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    lines = [f"# Benchmark: {meta['suite']} ({meta['timestamp']})", "",
             f"Machine: {meta['machine'].get('platform')} · {meta['machine'].get('cpu_count')} logical CPUs · "
             f"GPU: {meta['machine'].get('gpu') or 'none'} · time limit {meta['time_limit']} s · "
             f"HiGHS run single-threaded", "",
             "| instance | rows | cols | nnz | int | reference | ours | HiGHS | rel. err | our time | HiGHS time | ratio | iters / nodes | verdict |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        o, h = r["ours"], r.get("highs") or {}
        ratio = (o.get("time") / h["time"]) if h.get("time") and o.get("time") and r["ok"] else None
        errs = [e for e in (r.get("rel_error"), r.get("rel_error_vs_highs")) if e is not None]
        err = min(errs) if errs else None
        lines.append(
            f"| {r['name']} | {r['rows']} | {r['cols']} | {r['nnz']} | {r['int_vars']} | {_fmt_o(r.get('reference'))} | "
            f"{_fmt_o(o.get('objective')) if o.get('status') not in ('infeasible',) else 'infeasible'} | "
            f"{_fmt_o(h.get('objective')) if h.get('status') not in ('infeasible',) else 'infeasible'} | "
            f"{'-' if err is None else f'{err:.1e}'} | {_fmt_t(o.get('time'))} | {_fmt_t(h.get('time'))} | "
            f"{'-' if ratio is None else f'{ratio:.1f}x'} | {o.get('iterations', '-')} / {o.get('nodes', '-')} | {r['verdict']} |")
    ok = sum(r["ok"] for r in rows)
    lines += ["", f"**{ok} / {len(rows)} instances solved correctly.**"]
    notes = [(r["name"], r["note"]) for r in rows if r.get("note")]
    if notes:
        lines += ["", "Notes:"] + [f"- **{n}**: {t}" for n, t in notes]
    return "\n".join(lines) + "\n"


def size_of(inst: Instance) -> Dict[str, int]:
    from sovereign_opt.parsers.mps_parser import MPSParser
    m = MPSParser.parse_file(mps_path(inst))
    A = m.to_matrix_form()[0]
    q = m.get_quadratic_matrix()
    ints = sum(1 for v in m.variables.values() if v.var_type.value != "continuous")
    return {"rows": m.num_constraints, "cols": m.num_variables, "nnz": int(A.nnz), "int_vars": ints,
            "q_nnz": int(q.nnz), "problem_class": m.classify()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="quick", choices=sorted(SUITES))
    ap.add_argument("--instances", nargs="*", help="explicit keys, e.g. netlib:afiro miplib:flugpl")
    ap.add_argument("--algorithm", default="auto")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--no-highs", action="store_true")
    ap.add_argument("--no-presolve", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    args = ap.parse_args(argv)

    keys = args.instances or SUITES[args.suite]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    meta = {"suite": args.suite if not args.instances else "custom", "timestamp": stamp, "time_limit": args.time_limit,
            "algorithm": args.algorithm, "machine": machine_info()}
    rows = []
    hard = args.time_limit * 1.5 + 60
    options = {"enable_presolve": not args.no_presolve}
    for key in keys:
        inst = get_instance(key)
        if not os.path.exists(inst.path):
            print(f"{key:28s} missing file (run python -m benchmarks.fetch)")
            continue
        size = size_of(inst)
        is_mip = size["int_vars"] > 0
        highs = None if args.no_highs else _run(_highs, (key, args.time_limit), hard)
        ours = _run(_ours, (key, args.algorithm, args.time_limit, options), hard)
        v = verdict(inst, ours, highs, is_mip)
        row = {"key": key, "name": inst.name, "collection": inst.collection, "tags": inst.tags, "note": NOTES.get(inst.name),
               "published_optimum": inst.published_optimum, **size, "ours": ours, "highs": highs, **v}
        rows.append(row)
        print(f"{key:28s} {size['rows']:>7d}x{size['cols']:<7d} ours {str(ours.get('status')):12s} "
              f"{_fmt_o(ours.get('objective')):>18s} {_fmt_t(ours.get('time')):>9s} | HiGHS "
              f"{str((highs or {}).get('status')):10s} {_fmt_t((highs or {}).get('time')):>9s} | {v['verdict']}", flush=True)
        if ours.get("error"):
            print("    ", ours["error"])

    os.makedirs(RESULTS, exist_ok=True)
    base = os.path.join(RESULTS, f"{meta['suite']}{('_' + args.tag) if args.tag else ''}_{stamp}")
    with open(base + ".json", "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1, default=str)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(markdown(rows, meta))
    print(f"\n{sum(r['ok'] for r in rows)} / {len(rows)} correct. Results: {base}.json / .md")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
