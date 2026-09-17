"""
Tests for the benchmark tooling and the v3 engine additions: Netlib EMPS decoding, MPS writer,
PDLP (CPU / PyTorch backends), hybrid PDLP + crossover, concurrent racing, and the rule that the
comparison solver is never part of the sovereign engine.
"""
import os
import re

import numpy as np
import pytest

from benchmarks.industrial.power_dispatch import build_unit_commitment_model
from benchmarks.industrial.refinery_blending import build_refinery_blending_model
from benchmarks.industrial.supply_chain import build_supply_chain_arrays, build_supply_chain_model
from benchmarks.netlib.afiro import AFIRO_OPTIMAL_OBJECTIVE, build_netlib_afiro
from sovereign_opt.parsers.emps import expand_emps, is_emps
from sovereign_opt.parsers.mps_parser import MPSParser
from sovereign_opt.parsers.mps_writer import write_mps
from sovereign_opt.solvers.dispatch import solve_model
from sovereign_opt.solvers.lp.pdlp import pdlp

# The original compressed file from netlib.org/lp/data/afiro (794 bytes).
AFIRO_EMPS = '\nNAME          AFIRO\n       28       32        4       88        1        7        0        0\n        0        0        9\nhRMiR"/PV&QQ"2QQ"1QQ$;QQ$GPU)PS/\nER09\nER10\nLX05\nLX21\nER12\nER13\nLX17\nLX18\nLX19\nLX20\nER19\nER20\nLX27\nLX44\nER22\nER23\nLX40\nLX41\nLX42\nLX43\nLX45\nLX46\nLX47\nLX48\nLX49\nLX50\nLX51\nNCOST\n8X01\n!i?9z;;<c8X02\n=z9c!mhS%8X03\nOz9c8X04\n!kc;c8X06\n!j?>z?;@c8X07\n!j@>z?;Ac8X08\n!j@>z?iR"%Bc8X09\n!jQQ$T>z?hRyCc8X10\nNQQ;c@z8X11\nNQQ;yAz8X12\nNQQ<1Bz8X13\nNQQ<GCz8X14\n=B>c!mhRB8X15\n!hz>c8X16\n!lc?c8X22\nO=DzE9Fc8X23\nGzDc!mhS\'8X24\n!izDc8X25\nNzDc8X26\n!kcEc8X28\n!h=H9IcJc8X29\n!h>H9IcKc8X30\n!h>HhRIIcLc8X31\n!hQQ"0HhRGIcMc8X32\nNQQ8nJz8X33\nNQQ9,Kz8X34\nNQQ9KLz8X35\nNQQ9jMz8X36\nGBIz!mhRR8X37\n!jzIc8X38\n!lcHc8X39\nIc!mPU"\n8B\n!kPUA!lPV$<A@AF<I[}J<\n m0~TQ"#_`aP)(WU*+QRSTVWXYZYZZ(u.2MU%(^J"STxJg6$(gmr0JcIA;D`Mw[U9\n'


def test_emps_decoder_reproduces_afiro():
    assert is_emps(AFIRO_EMPS)
    model = MPSParser.parse_string(AFIRO_EMPS)
    assert model.num_variables == 32 and model.num_constraints == 27
    out = solve_model(model, algorithm="dual_simplex")
    assert abs(out.result.objective_value - AFIRO_OPTIMAL_OBJECTIVE) <= 1e-8 * abs(AFIRO_OPTIMAL_OBJECTIVE)
    assert out.certificate.status == "OPTIMAL_CERTIFIED"
    # decoded text parses the same as the embedded uncompressed MPS
    assert "ENDATA" in expand_emps(AFIRO_EMPS)


@pytest.mark.parametrize("build", [
    build_netlib_afiro,
    lambda: build_refinery_blending_model(True),
    lambda: build_unit_commitment_model(4),
])
def test_mps_writer_round_trip(build):
    model = build()
    again = MPSParser.parse_string(write_mps(model))
    a = solve_model(model).result.objective_value
    b = solve_model(again).result.objective_value
    assert abs(a - b) <= 1e-7 * max(1.0, abs(a))


def test_pdlp_cpu_reaches_netlib_optimum():
    m = build_netlib_afiro()
    A, rl, ru, cl, cu = m.to_matrix_form()
    r = pdlp(A, m.get_objective_vector(), rl, ru, cl, cu, tol=1e-7, device="cpu")
    assert r.status == "optimal"
    assert abs(r.primal_objective - AFIRO_OPTIMAL_OBJECTIVE) <= 1e-5 * abs(AFIRO_OPTIMAL_OBJECTIVE)


def test_pdlp_torch_backend_matches_numpy():
    pytest.importorskip("torch")
    m = build_netlib_afiro()
    A, rl, ru, cl, cu = m.to_matrix_form()
    c = m.get_objective_vector()
    a = pdlp(A, c, rl, ru, cl, cu, tol=1e-6, device="cpu")
    b = pdlp(A, c, rl, ru, cl, cu, tol=1e-6, device="torch-cpu")
    assert a.status == b.status == "optimal"
    assert abs(a.primal_objective - b.primal_objective) <= 1e-4


def test_hybrid_pdlp_is_certified_exact():
    out = solve_model(build_netlib_afiro(), algorithm="hybrid_pdlp")
    assert out.certificate.status == "OPTIMAL_CERTIFIED"
    assert abs(out.result.objective_value - AFIRO_OPTIMAL_OBJECTIVE) <= 1e-8 * abs(AFIRO_OPTIMAL_OBJECTIVE)


def test_supply_chain_generator_is_feasible_and_scales():
    lp = build_supply_chain_arrays(4, 20, 400)
    assert lp.A.shape[1] == 4 * 20 * 4 + 400 * 5 * 4
    out = solve_model(build_supply_chain_model(1500), algorithm="dual_simplex")
    assert out.certificate.status == "OPTIMAL_CERTIFIED"


def test_concurrent_race_returns_certified_answer():
    out = solve_model(build_refinery_blending_model(False), algorithm="concurrent", time_limit_seconds=60)
    assert out.certificate.status == "OPTIMAL_CERTIFIED"
    assert out.result.diagnostics["concurrent_winner"]
    assert abs(out.result.objective_value - (-6094333.333333333)) <= 1e-6 * 6094333.3


def test_lp_fallback_to_dual_simplex_when_method_stalls():
    from sovereign_opt.solvers import dispatch
    from sovereign_opt.solvers.base import SolverResult, SolverStatus

    class Stalls:
        name = "Stalls"

        def solve(self, model, **kw):
            return SolverResult(status=SolverStatus.ITERATION_LIMIT, iterations=200)

    original = dispatch.make_solver
    dispatch.make_solver = lambda alg, t: Stalls() if alg == "interior_point" else original(alg, t)
    try:
        out = dispatch.solve_model(build_netlib_afiro(), algorithm="interior_point")
    finally:
        dispatch.make_solver = original
    assert out.algorithm == "dual_simplex"
    assert out.certificate.status == "OPTIMAL_CERTIFIED"


def test_engine_never_imports_comparison_solvers():
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sovereign_opt")
    banned = re.compile(r"^\s*(import|from)\s+(highspy|gurobipy|cplex|pyscipopt|ortools|cvxpy|pulp)\b|linprog|scipy\.optimize", re.M)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                text = open(os.path.join(dirpath, f), encoding="utf-8").read()
                assert not banned.search(text), f"external solver referenced in {f}"
