"""Tests for the concept-drift seismograph UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_stream


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/run",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_stream_endpoint_runs_real_pipeline(server):
    status, payload = _post(server, {"data_size": 400, "drift_at": 200, "detector": "both"})
    assert status == 200
    assert payload["data_size"] == 400
    assert payload["warmup_size"] > 0
    assert set(payload["runs"].keys()) == {"adwin", "page_hinkley"}
    # events serialize a 'step' key so the timeline can mark them
    for run in payload["runs"].values():
        for event in run["events"]:
            assert "step" in event


def test_adaptive_beats_static_after_drift(server):
    status, payload = _post(server, {"data_size": 600, "drift_at": 300, "detector": "both"})
    assert status == 200
    assert payload["adaptive"]["post_drift_mae"] < payload["static"]["post_drift_mae"]


def test_health_and_index_served(server):
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    assert health["status"] == "ok" and health["detectors"] == ["adwin", "page_hinkley", "both"]
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "seismograph" in html and "/api/run" in html


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"data_size": "x"}, "data_size must be an integer"),
        ({"data_size": 100, "drift_at": True}, "drift_at must be an integer"),
        ({"drift_type": "slow"}, "drift_type must be 'abrupt' or 'gradual'"),
        ({"detector": "magic"}, "detector must be one of"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    status, body = _post(server, payload)
    assert status == 400
    assert message in body["error"]


def test_run_stream_rejects_bad_type_directly():
    with pytest.raises(ValueError, match="data_size must be an integer"):
        run_stream(data_size="x")