"""
Numerical robustness report.

    python -m benchmarks.robustness
    python -m benchmarks.robustness --time-limit 300

Groups the hard instances by WHY they are hard and shows how the engine copes:
- degenerate          many ties in the ratio test -> cycling/stalling risk
                      (counters: degenerate pivots, bound perturbations, Bland anti-cycling pivots)
- ill-conditioned     coefficient ranges over many orders of magnitude -> round-off, singular bases
                      (coefficient range before / after our scaling, basis repairs, numerical recoveries),
                      including synthetic stress copies of Netlib LPs rescaled by random powers of ten
- weak LP relaxation  MILPs whose LP bound is far from the integer optimum -> huge trees
                      (root bound, final gap, nodes)
- infeasible          must be PROVEN infeasible, not just "not found"
Every answer is still checked against the published optimum and HiGHS, and certified independently.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from benchmarks.compare import RESULTS, _highs, _ours, _run, machine_info, verdict
from benchmarks.instances import SUITES, get_instance, mps_path

GROUPS = [("degenerate", "Degenerate LPs"), ("ill-conditioned", "Ill-conditioned / badly scaled LPs"),
          ("weak-relaxation", "MILPs with weak LP relaxations"), ("big-M", "MILPs with weak LP relaxations"),
          ("infeasible", "Infeasible LPs (must be proven)")]


def coefficient_stats(path: str):
    from sovereign_opt.parsers.mps_parser import MPSParser
    from sovereign_opt.sparse.scaling import geometric_scaling
    import scipy.sparse as sp
    m = MPSParser.parse_file(path)
    A = sp.csr_matrix(m.to_matrix_form()[0])
    a = np.abs(A.data[A.data != 0])
    if a.size == 0:
        return m, {}
    R, C = geometric_scaling(A)
    S = sp.csr_matrix(sp.diags(R) @ A @ sp.diags(C))
    s = np.abs(S.data[S.data != 0])
    return m, {"coef_min": float(a.min()), "coef_max": float(a.max()), "range_before": float(a.max() / a.min()),
               "range_after": float(s.max() / s.min())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--algorithm", default="auto")
    args = ap.parse_args(argv)

    rows = []
    hard = args.time_limit * 1.5 + 60
    for key in SUITES["robust"]:
        inst = get_instance(key)
        if not os.path.exists(inst.path):
            print(f"{key}: missing data (python -m benchmarks.fetch)")
            continue
        model, coef = coefficient_stats(mps_path(inst))
        is_mip = any(v.var_type.value != "continuous" for v in model.variables.values())
        highs = _run(_highs, (key, args.time_limit), hard)
        ours = _run(_ours, (key, args.algorithm, args.time_limit, {}), hard)
        v = verdict(inst, ours, highs, is_mip)
        d = ours.get("diagnostics", {})
        row = {"key": key, "name": inst.name, "tags": inst.tags, "rows": model.num_constraints,
               "cols": model.num_variables, "published_optimum": inst.published_optimum, **coef,
               "ours": ours, "highs": highs, **v}
        rows.append(row)
        print(f"{key:24s} {model.num_constraints:>6d}x{model.num_variables:<6d} range {coef.get('range_before', 0):9.1e}"
              f" -> {coef.get('range_after', 0):8.1e} | {ours.get('status'):12s} {ours.get('time', 0):7.2f}s | "
              f"degen {d.get('degenerate_pivots', '-')} perturb {d.get('bound_perturbations', '-')} bland "
              f"{d.get('bland_pivots', '-')} repairs {d.get('basis_repairs', '-')} | {v['verdict']}", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(RESULTS, exist_ok=True)
    base = os.path.join(RESULTS, f"robustness_{stamp}")
    meta = {"timestamp": stamp, "machine": machine_info(), "time_limit": args.time_limit}
    with open(base + ".json", "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1, default=str)

    md = [f"# Numerical robustness report ({stamp})", "",
          f"Time limit {args.time_limit:.0f} s per solve · {meta['machine'].get('cpu_count')} logical CPUs", ""]
    done = set()
    for tag, title in GROUPS:
        group = [r for r in rows if tag in r["tags"] and r["key"] not in done]
        if not group:
            continue
        done.update(r["key"] for r in group)
        md += [f"## {title}", "",
               "| instance | size | coef. range (raw -> scaled) | ours | HiGHS | rel. err | time (ours / HiGHS) | method | degenerate pivots | perturbations | Bland pivots | basis repairs | nodes | verdict |",
               "|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|"]
        for r in group:
            o, h, d = r["ours"], r["highs"] or {}, r["ours"].get("diagnostics", {})
            err = r.get("rel_error") if r.get("rel_error") is not None else r.get("rel_error_vs_highs")
            md.append(
                f"| {r['name']} | {r['rows']}x{r['cols']} | {r.get('range_before', 0):.1e} -> {r.get('range_after', 0):.1e} | "
                f"{o.get('status')} {'' if o.get('objective') is None else f'{o['objective']:.8g}'} | "
                f"{h.get('status')} {'' if h.get('objective') is None else f'{h['objective']:.8g}'} | "
                f"{'-' if err is None else f'{err:.1e}'} | {o.get('time', 0):.2f} s / {h.get('time', 0):.2f} s | "
                f"{o.get('algorithm', '-')} | {d.get('degenerate_pivots', '-')} | {d.get('bound_perturbations', '-')} | "
                f"{d.get('bland_pivots', '-')} | {d.get('basis_repairs', '-')} | {o.get('nodes', '-')} | {r['verdict']} |")
        md.append("")
    ok = sum(r["ok"] for r in rows)
    md.append(f"**{ok} / {len(rows)} handled correctly.**")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n{ok} / {len(rows)} correct. Report: {base}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
