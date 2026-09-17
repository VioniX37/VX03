"""
Train the LP algorithm selector from measured runs (the "learned" part of the strategy engine).

    python -m benchmarks.train_selector                 # measure + train
    python -m benchmarks.train_selector --train-only    # retrain from the saved dataset

1. Measure: every LP in the training set is solved with each candidate algorithm (own process,
   time limit). A run counts only if it is certified optimal and matches the published optimum.
2. Label: the fastest correct algorithm for that instance.
3. Train: multinomial logistic regression (NumPy, L2-regularised, standardised features) on the
   structural features from sovereign_opt/ml/features.py.
4. Validate honestly with leave-one-out cross-validation, reporting both accuracy and the "regret"
   (time of the chosen algorithm / time of the best algorithm) against always using one method.
5. Save weights to sovereign_opt/ml/lp_selector.json, which MLStrategyEngine loads at runtime.
   The hand-written rules remain as the fallback when the file is missing or confidence is low.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from benchmarks.compare import _ours, _run
from benchmarks.instances import KENNINGTON, NETLIB_LARGE, NETLIB_SMALL, get_instance, mps_path

ALGOS = ["dual_simplex", "simplex", "interior_point", "hybrid_pdlp"]
DATASET = os.path.join(os.path.dirname(__file__), "results", "selector_dataset.json")
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sovereign_opt", "ml", "lp_selector.json")
FEATURES = ["log_num_vars", "log_num_cons", "log_num_nnz", "density", "aspect_ratio", "coeff_range_log10",
            "frac_boxed", "frac_eq_cons", "frac_le_cons", "est_condition_log10"]


def measure(time_limit: float):
    from sovereign_opt.ml.features import FeatureExtractor
    from sovereign_opt.parsers.mps_parser import MPSParser
    data = json.load(open(DATASET)) if os.path.exists(DATASET) else {}
    keys = [f"netlib:{n}" for n in NETLIB_SMALL + NETLIB_LARGE] + [f"kennington:{n}" for n in KENNINGTON]
    for key in keys:
        inst = get_instance(key)
        if not os.path.exists(inst.path) or key in data:
            continue
        model = MPSParser.parse_file(mps_path(inst))
        feats = FeatureExtractor.extract(model)
        entry = {"features": {k: float(feats[k]) for k in FEATURES}, "runs": {}}
        for algo in ALGOS:
            r = _run(_ours, (key, algo, time_limit, {}), time_limit * 1.5 + 60)
            ok = (r.get("certificate") == "OPTIMAL_CERTIFIED" and r.get("objective") is not None
                  and inst.published_optimum is not None
                  and abs(r["objective"] - inst.published_optimum) / max(1.0, abs(inst.published_optimum)) <= 1e-6)
            fell_back = "fallback_from" in json.dumps(r.get("diagnostics", {})) or r.get("algorithm") != algo
            entry["runs"][algo] = {"ok": bool(ok and not fell_back), "time": r.get("time"), "status": r.get("status")}
            print(f"{key:24s} {algo:16s} {r.get('status')!s:14s} {r.get('time', 0):8.2f}s ok={ok and not fell_back}", flush=True)
        data[key] = entry
        os.makedirs(os.path.dirname(DATASET), exist_ok=True)
        json.dump(data, open(DATASET, "w"), indent=1)
    return data


def _fit(X, y, k, lam=0.3, iters=3000, lr=0.2):
    n, d = X.shape
    W = np.zeros((k, d + 1))
    Xb = np.hstack([X, np.ones((n, 1))])
    Y = np.eye(k)[y]
    for _ in range(iters):
        Z = Xb @ W.T
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y).T @ Xb / n
        G[:, :-1] += lam * W[:, :-1]
        W -= lr * G
    return W


def train(data):
    names, X, y, times = [], [], [], []
    for key, e in data.items():
        runs = e["runs"]
        good = {a: r["time"] for a, r in runs.items() if r["ok"] and r["time"]}
        if not good:
            continue
        names.append(key)
        X.append([e["features"][f] for f in FEATURES])
        y.append(ALGOS.index(min(good, key=good.get)))
        times.append([runs[a]["time"] if runs[a]["ok"] and runs[a]["time"] else np.inf for a in ALGOS])
    X, y, T = np.array(X), np.array(y), np.array(times)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    k = len(ALGOS)

    # leave-one-out
    pred = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        W = _fit(Xs[mask], y[mask], k)
        pred[i] = int(np.argmax(W @ np.append(Xs[i], 1.0)))
    best = T.min(axis=1)
    PEN = 10.0  # a wrong / failed pick costs 10x the best time (fallback path)

    def regret(choice):
        t = np.array([T[i, c] if np.isfinite(T[i, c]) else PEN * best[i] for i, c in enumerate(choice)])
        return float(np.exp(np.mean(np.log(t / best))))

    report = {"instances": len(y), "loo_accuracy": float((pred == y).mean()),
              "loo_time_regret_learned": regret(pred),
              "label_counts": {ALGOS[j]: int((y == j).sum()) for j in range(k)}}
    for j, a in enumerate(ALGOS):
        report[f"time_regret_always_{a}"] = regret(np.full(len(y), j))
    W = _fit(Xs, y, k)
    out = {"algorithms": ALGOS, "features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(), "weights": W.tolist(),
           "trained_on": names, "validation": report, "created": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(out, open(WEIGHTS, "w"), indent=1)
    print(json.dumps(report, indent=1))
    print("saved", WEIGHTS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args(argv)
    data = json.load(open(DATASET)) if args.train_only else measure(args.time_limit)
    train(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
