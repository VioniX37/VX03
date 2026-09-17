"""
Package the repo (code + downloaded benchmark data) for upload as a Kaggle dataset.

    python tools/make_kaggle_bundle.py   ->  dist/sovereign_opt_bundle.zip
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = ["sovereign_opt", "benchmarks", "tests", "requirements.txt"]
SKIP_DIRS = {"__pycache__", "cache", "node_modules", ".next", "results"}


def main():
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", "sovereign_opt_bundle.zip")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = os.path.join(ROOT, item)
            if os.path.isfile(path):
                z.write(path, item)
                n += 1
                continue
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for f in filenames:
                    full = os.path.join(dirpath, f)
                    z.write(full, os.path.relpath(full, ROOT))
                    n += 1
    print(f"{out}  ({n} files, {os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
