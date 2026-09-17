"""
Demo race: registry integrity, license handling, and the API surface the demo page depends on.

These are cheap checks -- nothing here actually races a solver. The point is that the
presentation path cannot silently break: the registry stays consistent with what the page
renders, an oversized model is reported rather than attempted on a capped license, and a
saved demo result never corrupts the rest of the server.
"""
import json
import os

import pytest

from benchmarks.demo_race import (
    DEMO_INSTANCES,
    DEMO_ORDER,
    GUROBI_FREE_ROWS,
    GUROBI_FREE_VARS,
    RESULTS,
    SOLVER_LABELS,
    instance_payload,
    race_one,
    solver_applicable,
)

LIMITED = {"available": True, "size_limited": True, "version": "13.0.3"}
UNLIMITED = {"available": False, "size_limited": None, "note": "gurobipy is not installed"}
FULL = {"available": True, "size_limited": False, "version": "13.0.3"}


def test_four_visible_instances_in_presentation_order():
    assert DEMO_ORDER == ["supply_chain_100k", "supply_chain_1m", "qp_stcqp1", "milp_uc_720"]
    assert [DEMO_INSTANCES[i].letter for i in DEMO_ORDER] == ["A", "B", "C", "D"]


@pytest.mark.parametrize("instance_id", list(DEMO_INSTANCES))
def test_registry_entry_is_renderable(instance_id):
    """Everything the demo page reads must be present before any solver has run."""
    inst = DEMO_INSTANCES[instance_id]
    p = instance_payload(inst)
    assert p["size"]["rows"] > 0 and p["size"]["cols"] > 0 and p["size"]["nnz"] > 0
    assert inst.problem_class in {"LP", "QP", "MILP"}
    assert {s["key"] for s in p["solvers"]} <= set(SOLVER_LABELS)
    # a recorded fallback for every solver, so a failed live run still draws a bar
    for key in inst.solvers:
        rec = inst.recorded.get(key)
        assert isinstance(rec, dict), f"{instance_id}: no recorded result for {key}"
        assert rec["time"] > 0 and rec["objective"] is not None


@pytest.mark.parametrize("instance_id", DEMO_ORDER)
def test_recorded_objectives_agree_across_solvers(instance_id):
    """The numbers shipped in the registry must themselves pass the match test we show on stage."""
    inst = DEMO_INSTANCES[instance_id]
    objs = [r["objective"] for k, r in inst.recorded.items() if isinstance(r, dict)]
    ref = objs[0]
    tol = 1e-4 if inst.size.get("int_vars") else 1e-3
    for o in objs:
        assert abs(o - ref) / max(1.0, abs(ref)) <= tol, f"{instance_id}: recorded objectives disagree"


def test_size_limited_license_skips_the_big_models_only():
    big = DEMO_INSTANCES["supply_chain_1m"]
    small = DEMO_INSTANCES["milp_uc_720"]
    assert small.size["cols"] <= GUROBI_FREE_VARS and small.size["rows"] <= GUROBI_FREE_ROWS

    ok, reason = solver_applicable(big, "gurobi", LIMITED)
    assert not ok and "size-limited" in reason
    assert solver_applicable(small, "gurobi", LIMITED)[0]
    # a full license runs everything
    assert solver_applicable(big, "gurobi", FULL)[0]
    # and with no probe at all we do not pre-judge; the runner reports the truth
    assert solver_applicable(big, "gurobi", None)[0]


def test_missing_gurobi_is_reported_not_crashed():
    ok, reason = solver_applicable(DEMO_INSTANCES["milp_uc_720"], "gurobi", UNLIMITED)
    assert not ok and "not installed" in reason


def test_unknown_solver_is_skipped_without_building_the_model():
    """A skipped race is instant: no 1M-variable matrix gets generated just to say 'no'."""
    rec = race_one("supply_chain_1m", "nosuchsolver")
    assert rec["status"] == "skipped"
    assert "build_time" not in rec


def test_demo_result_files_do_not_break_the_server(tmp_path):
    """
    Regression: benchmarks/results/demo_*.json store a single record whose "rows" is a row
    count, not a list of benchmark rows. The preset scanner must skip them.
    """
    from sovereign_opt.server import _register_benchmark_library

    os.makedirs(RESULTS, exist_ok=True)
    probe = os.path.join(RESULTS, "demo_pytest_probe_20200101-000000.json")
    with open(probe, "w") as f:
        json.dump({"instance": "x", "solver": "ours", "rows": 1228, "cols": 720}, f)
    try:
        _register_benchmark_library()  # must not raise
    finally:
        os.remove(probe)


def test_api_lists_instances_and_replays_recorded():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from sovereign_opt.server import app

    client = TestClient(app)
    listing = client.get("/api/demo/instances")
    assert listing.status_code == 200
    visible = [i for i in listing.json() if not i["hidden"]]
    assert [i["letter"] for i in visible] == ["A", "B", "C", "D"]

    # the presenter's parachute: a complete race, instantly, with no solver process at all
    res = client.post("/api/demo/race", json={"instance_id": "supply_chain_1m", "use_recorded": True})
    assert res.status_code == 200
    job = res.json()
    assert job["state"] == "done" and job["source"] == "recorded"
    assert job["solvers"]["ours"]["solve_time"] > 0
    assert job["speedup"]["vs_highs"] > 1  # we are faster than HiGHS at a million variables
    assert job["speedup"]["vs_gurobi"] > 1  # and faster than Gurobi

    assert client.post("/api/demo/race", json={"instance_id": "nope"}).status_code == 404
    assert client.get("/api/demo/race/deadbeef").status_code == 404
