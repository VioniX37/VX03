"""
Download the public benchmark instances into benchmarks/data/ (not committed to git).

    python -m benchmarks.fetch                # everything (~15 MB transfer)
    python -m benchmarks.fetch --only netlib miplib

Sources (official collection sites):
    Netlib LP            https://www.netlib.org/lp/data/            (compressed EMPS; expanded by our parser)
    Netlib Kennington    https://www.netlib.org/lp/data/kennington/
    Netlib infeasible    https://www.netlib.org/lp/infeas/
    MIPLIB 2017          https://miplib.zib.de/WebData/instances/   (+ miplib2017-v31.solu optimal values)
    Maros-Meszaros QP    https://www.doc.ic.ac.uk/~im/QPDATA1.ZIP   (only the instances we use are kept)
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

from benchmarks.instances import DATA, INFEASIBLE, KENNINGTON, MIPLIB, NETLIB_LARGE, NETLIB_SMALL, QP_LARGE, QP_SMALL

NETLIB = "https://www.netlib.org/lp/data/"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sovereign-opt-benchmarks"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _save(path: str, url: str):
    if os.path.exists(path) and os.path.getsize(path) > 400:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        data = _get(url)
        with open(path, "wb") as f:
            f.write(data)
        print(f"  {os.path.relpath(path, DATA):40s} {len(data) / 1024:8.1f} KB")
    except Exception as exc:
        print(f"  FAILED {url}: {exc}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["netlib", "infeas", "miplib", "qp"])
    args = ap.parse_args(argv)
    if "netlib" in args.only:
        print("Netlib LP")
        _save(os.path.join(DATA, "netlib", "readme"), NETLIB + "readme")
        for n in NETLIB_SMALL + NETLIB_LARGE:
            _save(os.path.join(DATA, "netlib", n), NETLIB + n)
        for n in KENNINGTON:
            _save(os.path.join(DATA, "netlib", n + ".gz"), NETLIB + "kennington/" + n + ".gz")
    if "infeas" in args.only:
        print("Netlib infeasible")
        for n in INFEASIBLE:
            _save(os.path.join(DATA, "infeas", n), "https://www.netlib.org/lp/infeas/" + n)
    if "miplib" in args.only:
        print("MIPLIB 2017")
        _save(os.path.join(DATA, "miplib", "miplib2017-v31.solu"), "https://miplib.zib.de/downloads/miplib2017-v31.solu")
        for n in MIPLIB:
            _save(os.path.join(DATA, "miplib", n + ".mps.gz"), f"https://miplib.zib.de/WebData/instances/{n}.mps.gz")
    if "qp" in args.only:
        print("Maros-Meszaros QP")
        wanted = [f"{n}.QPS" for n in QP_SMALL + QP_LARGE]
        qp_dir = os.path.join(DATA, "qp")
        if not all(os.path.exists(os.path.join(qp_dir, w)) for w in wanted):
            z = zipfile.ZipFile(io.BytesIO(_get("https://www.doc.ic.ac.uk/~im/QPDATA1.ZIP")))
            os.makedirs(qp_dir, exist_ok=True)
            for w in wanted:
                with open(os.path.join(qp_dir, w), "wb") as f:
                    f.write(z.read(w))
            print(f"  extracted {len(wanted)} QPS files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
