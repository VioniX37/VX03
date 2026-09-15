"""
REST API tests: every preset runs end-to-end and produces JSON-safe, certified responses.
"""
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from sovereign_opt.server import app, PRESETS

client = TestClient(app)


@pytest.mark.parametrize("preset_id", list(PRESETS))
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


def test_unknown_algorithm_is_rejected():
    client.post("/api/load_preset", json={"preset_id": "netlib_afiro"})
    assert client.post("/api/solve", json={"algorithm": "magic"}).status_code == 400


def test_incompatible_algorithm_is_redirected_with_note():
    client.post("/api/load_preset", json={"preset_id": "power_unit_commitment"})
    body = client.post("/api/solve", json={"algorithm": "simplex"}).json()
    assert body["algorithm_key"] == "branch_and_bound"
    assert body["notes"]
