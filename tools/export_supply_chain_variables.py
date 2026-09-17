"""
Export every variable of the production-distribution network LP to a CSV file.

    python tools/export_supply_chain_variables.py                 # 50k preset -> exports/supply_chain_50k_variables.csv
    python tools/export_supply_chain_variables.py --target 3000   # the small preset

Columns:
    index         column position in the model (the UI shows it as v<index>)
    ui_name       name shown in the dashboard
    readable_name what the variable means
    kind          plant_to_warehouse (x) or warehouse_to_customer (z)
    plant, warehouse, customer, product   the route and product (blank where not applicable)
    lower_bound, upper_bound, cost_per_unit
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.industrial.supply_chain import build_supply_chain_arrays, sizes_for  # noqa: E402

PRODUCTS = ["diesel", "petrol", "lpg", "lubricant", "atf", "bitumen"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lp = build_supply_chain_arrays(*sizes_for(args.target))
    meta = lp.meta
    P, W, C, K, L = meta["plants"], meta["warehouses"], meta["customers"], meta["products"], meta["neighbours"]
    near = meta["near"]
    names = PRODUCTS[:K] if K <= len(PRODUCTS) else [f"product_{k}" for k in range(K)]
    out = args.out or os.path.join("exports", f"supply_chain_{round(args.target / 1000)}k_variables.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    header = ["index", "ui_name", "readable_name", "kind", "plant", "warehouse", "customer", "product",
              "lower_bound", "upper_bound", "cost_per_unit"]
    j = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for p in range(P):  # x[p, w, k], same order as the generator
            for wh in range(W):
                for k in range(K):
                    w.writerow([j, f"v{j}", f"ship_plant{p}_to_warehouse{wh}_{names[k]}", "plant_to_warehouse",
                                p, wh, "", names[k], lp.col_lb[j], "inf", round(float(lp.c[j]), 6)])
                    j += 1
        for c in range(C):  # z[c, l, k], l-th nearest warehouse of customer c
            for l in range(L):
                wh = int(near[c, l])
                for k in range(K):
                    w.writerow([j, f"v{j}", f"deliver_warehouse{wh}_to_customer{c}_{names[k]}",
                                "warehouse_to_customer", "", wh, c, names[k], lp.col_lb[j], "inf",
                                round(float(lp.c[j]), 6)])
                    j += 1
    assert j == lp.A.shape[1], (j, lp.A.shape)
    print(f"{out}: {j} variables ({P * W * K} plant->warehouse, {C * L * K} warehouse->customer), "
          f"{os.path.getsize(out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
