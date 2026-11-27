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

def test_index_has_accessibility_features(server):
    """The served page ships skip-link navigation, a main landmark, and reduced-motion guard."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'class="skip-link"' in html, "missing skip link for keyboard users"
    assert '<main class="layout">' in html, "missing main landmark"
    assert 'role="alert"' in html, "error box must announce to assistive tech"
    assert "prefers-reduced-motion" in html, "missing reduced-motion guard"


def test_index_has_loading_and_theme_states(server):
    """The page shows busy feedback during a run and declares its dark-first color scheme."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'aria-busy' in html, "form must announce busy state during a run"
    assert 'class="spinner"' in html, "missing visible busy indicator"
    assert "color-scheme: dark" in html, "dark-first page must declare its color scheme"


def test_oversized_declared_body_is_rejected_not_hung(server):
    """A lying Content-Length must get a fast 413, never park a handler thread."""
    import socket
    from urllib.parse import urlparse

    u = urlparse(server)
    with socket.create_connection((u.hostname, u.port), timeout=5) as sock:
        sock.sendall(
            b"POST /api/run HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
            b"Content-Length: 999999999\r\n\r\n{}"
        )
        first = sock.recv(4096).split(b"\r\n", 1)[0]
    assert b"413" in first, f"expected 413 for oversized body, got {first!r}"


def test_index_disables_heuristic_caching(server):
    """The UI shell must not be heuristically cached — stale HTML after edits was
    observed live because custom do_GET handlers send no cache validators."""
    resp = urllib.request.urlopen(server + "/")
    assert resp.headers.get("Cache-Control") == "no-cache"
