"""
GPU parity check (run on a CUDA machine, e.g. the Kaggle notebook):

    python -m benchmarks.gpu_parity

Solves real Netlib LPs with PDLP on CPU (NumPy) and on the GPU (PyTorch CUDA) and checks that both
reach the published optimum. Without a GPU it compares against the PyTorch CPU backend instead.
"""
import sys
import warnings

from benchmarks.instances import get_instance, mps_path
from sovereign_opt.parsers.mps_parser import MPSParser
from sovereign_opt.solvers.lp.pdlp import pdlp


def main() -> int:
    warnings.filterwarnings("ignore")
    try:
        import torch
        gpu = "cuda" if torch.cuda.is_available() else "torch-cpu"
    except ImportError:
        print("PyTorch not installed")
        return 1
    ok = True
    for name in ("afiro", "blend", "israel", "brandy"):
        inst = get_instance("netlib:" + name)
        m = MPSParser.parse_file(mps_path(inst))
        A, rl, ru, cl, cu = m.to_matrix_form()
        sign = 1.0 if m.objective.sense == "minimize" else -1.0
        c = sign * m.get_objective_vector()
        objs = {}
        for dev in ("cpu", gpu):
            r = pdlp(A, c, rl, ru, cl, cu, tol=1e-6, time_limit=120, device=dev)
            objs[dev] = sign * r.primal_objective + m.objective.offset
            err = abs(objs[dev] - inst.published_optimum) / max(1.0, abs(inst.published_optimum))
            ok &= r.status == "optimal" and err <= 1e-4
            print(f"{name:8s} {r.device:28s} {r.status:10s} obj {objs[dev]:16.8g} err {err:.1e} iters {r.iterations:6d} {r.runtime:.2f}s")
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
