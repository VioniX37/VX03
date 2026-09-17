"""
REST API tests: every preset runs end-to-end and produces JSON-safe, certified responses.
"""
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from sovereign_opt.server import app, PRESETS

client = TestClient(app)


LONG_RUNNING = {"supply_chain_lp_50k", "unit_commitment_fleet_2880", "facility_location_1220"}
DEMO_PRESETS = [p for p in PRESETS if "/" not in p and p not in LONG_RUNNING]


@pytest.mark.parametrize("preset_id", DEMO_PRESETS)
def test_every_preset_solves_and_certifies(preset_id):
    assert client.post("/api/load_preset", json={"preset_id": preset_id}).status_code == 200
    assert client.post("/api/presolve").status_code == 200
    assert client.post("/api/ml_recommend").status_code == 200
    res = client.post("/api/solve", json={"algorithm": "auto", "enable_presolve": True, "time_limit_seconds": 60})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "optimal"
    assert body["trust_certificate"]["status"] == "OPTIMAL_CERTIFIED"
    assert len(body["workflow_stages"]) == 6
    for node in body["diagnostics"].get("tree_trace", []):
        assert isinstance(node["lower_bound"], (int, float))


def test_benchmark_library_presets_are_listed():
    ids = [p["id"] for p in client.get("/api/presets").json()]
    assert "supply_chain_lp_50k" in ids
    library = [i for i in ids if "/" in i]
    if library:  # only when benchmark data has been downloaded
        assert client.post("/api/load_preset", json={"preset_id": library[0]}).status_code == 200


def test_unknown_algorithm_is_rejected():
    client.post("/api/load_preset", json={"preset_id": "netlib_afiro"})
    assert client.post("/api/solve", json={"algorithm": "magic"}).status_code == 400


def test_incompatible_algorithm_is_redirected_with_note():
    client.post("/api/load_preset", json={"preset_id": "power_unit_commitment"})
    body = client.post("/api/solve", json={"algorithm": "simplex"}).json()
    assert body["algorithm_key"] == "branch_and_bound"
    assert body["notes"]
