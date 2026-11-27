"""Lightweight local web UI server for the concept-drift detector.

Serves the stream seismograph in ``ui/index.html`` and exposes a JSON API that runs the
real ``DriftDetectorPipeline``. Standard library only. Start with ``python ui/server.py``
and open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import DriftDetectorPipeline  # noqa: E402
from src.utils import generate_concept_drift_stream  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

DETECTOR_MODES = ("adwin", "page_hinkley", "both")


def run_stream(
    data_size: int = 1000,
    drift_at: int = 500,
    shift_magnitude: float = 3.0,
    drift_type: str = "abrupt",
    detector: str = "both",
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a concept-drift stream and run the detector pipeline.

    Raises ValueError with a clear message for invalid parameters, matching the CLI's
    boundary validation.
    """
    if isinstance(data_size, bool) or not isinstance(data_size, int):
        raise ValueError(f"data_size must be an integer, got {data_size!r}")
    if isinstance(drift_at, bool) or not isinstance(drift_at, int):
        raise ValueError(f"drift_at must be an integer, got {drift_at!r}")
    if drift_type not in ("abrupt", "gradual"):
        raise ValueError(f"drift_type must be 'abrupt' or 'gradual', got {drift_type!r}")
    if detector not in DETECTOR_MODES:
        raise ValueError(f"detector must be one of {list(DETECTOR_MODES)}, got {detector!r}")

    warmup_size = min(100, max(20, drift_at // 2 if drift_at > 0 else data_size // 5))
    X, y = generate_concept_drift_stream(
        data_size=data_size,
        drift_at=drift_at,
        shift_magnitude=shift_magnitude,
        drift_type=drift_type,
        seed=seed,
    )
    pipeline = DriftDetectorPipeline(detector=detector, warmup_size=warmup_size, seed=seed)
    result = pipeline.run(X, y, drift_at=drift_at)
    payload = result.to_dict()
    payload["warmup_size"] = warmup_size
    return payload


MAX_REQUEST_BYTES = 4_000_000


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok", "detectors": list(DETECTOR_MODES)})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/run":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_REQUEST_BYTES:
                    self._send_json(
                        {"error": f"request body exceeds {MAX_REQUEST_BYTES} bytes"},
                        status=413,
                    )
                    return
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = run_stream(
                    data_size=payload.get("data_size", 1000),
                    drift_at=payload.get("drift_at", 500),
                    shift_magnitude=payload.get("shift_magnitude", 3.0),
                    drift_type=payload.get("drift_type", "abrupt"),
                    detector=payload.get("detector", "both"),
                    seed=payload.get("seed", 42),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the concept-drift detector UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8126)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"Concept-drift seismograph running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())