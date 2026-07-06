"""
server.py
=========
Pure-stdlib HTTP server for the VitalWatch patient deterioration system.
No Flask / FastAPI / uvicorn required — runs on Python 3.8+.

API
---
GET  /                    → frontend HTML
GET  /static/<path>       → static files
GET  /api/status          → training progress + summary
GET  /api/charts          → base64-encoded chart images
GET  /api/metrics         → full metrics table
GET  /api/features        → top feature importances
POST /api/predict         → single-patient prediction (JSON body)
POST /api/batch           → CSV batch prediction    (raw text/csv body)
"""

import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Add backend dir to path so ml_engine is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import ml_engine as ml  # noqa: E402

# ---------------------------------------------------------------------------
# Training state (module-level)
# ---------------------------------------------------------------------------
_STATE:    dict  = {}
_READY:    bool  = False
_ERROR:    str   = ""
_PROGRESS: dict  = {"stage": "Initialising...", "pct": 0}

_FRONTEND = os.path.join(os.path.dirname(_HERE), "frontend")


# ---------------------------------------------------------------------------
# Background training thread
# ---------------------------------------------------------------------------

def _train():
    global _STATE, _READY, _ERROR, _PROGRESS
    stages = [
        ("Generating synthetic patient dataset (1,500 patients × 12 obs)...", 5),
        ("Engineering 10 feature families (33 features total)...",             20),
        ("Computing NEWS2 scores for every patient...",                         30),
        ("Applying SMOTE oversampling to correct class imbalance...",           42),
        ("Training Logistic Regression (L2, balanced)...",                      55),
        ("Training Random Forest (200 trees, balanced)...",                     68),
        ("Training Gradient Boosting (200 estimators)...",                      80),
        ("Computing permutation feature importances (20 repeats)...",           91),
        ("Generating analytical charts...",                                     97),
        ("Finalising deployment state...",                                      99),
    ]
    try:
        for stage, pct in stages:
            _PROGRESS = {"stage": stage, "pct": pct}
            time.sleep(0.25)

        _STATE = ml.run_pipeline()
        _PROGRESS = {"stage": "Training complete!", "pct": 100}
        _READY = True

    except Exception as exc:
        import traceback
        _ERROR    = f"{exc}\n{traceback.format_exc()}"
        _PROGRESS = {"stage": f"ERROR: {exc}", "pct": 0}


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def log_message(self, *_):
        pass  # silence default logging

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data, code: int = 200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    # ── CORS preflight ───────────────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── GET ──────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")

        # ── API routes
        if path == "/api/status":
            summary = {}
            if _READY:
                best = _STATE["best_name"]
                summary = {
                    "n_patients":   _STATE["n_patients"],
                    "n_features":   _STATE["n_features"],
                    "best_model":   best,
                    "best_auroc":   round(_STATE["results"][best]["auroc"], 4),
                    "prevalence":   round(_STATE["prevalence"] * 100, 1),
                    "smote_before": _STATE["smote_before"],
                    "smote_after":  _STATE["smote_after"],
                }
            self._send_json({
                "ready":    _READY,
                "error":    _ERROR,
                "progress": _PROGRESS,
                "summary":  summary,
            })
            return

        if path == "/api/charts":
            if not _READY:
                self._send_json({"error": "Not ready"}, 503); return
            self._send_json({"charts": _STATE["charts"]})
            return

        if path == "/api/metrics":
            if not _READY:
                self._send_json({"error": "Not ready"}, 503); return
            self._send_json({"metrics": _STATE["metrics"]})
            return

        if path == "/api/features":
            if not _READY:
                self._send_json({"error": "Not ready"}, 503); return
            self._send_json({
                "top_features":   _STATE["top_features"],
                "total_features": _STATE["n_features"],
            })
            return

        # ── Static / HTML
        if path in ("", "/", ""):
            self._serve(os.path.join(_FRONTEND, "index.html"))
            return

        rel      = path.lstrip("/")
        filepath = os.path.join(_FRONTEND, rel)
        if os.path.isfile(filepath):
            self._serve(filepath)
            return

        self._send_json({"error": f"Not found: {path}"}, 404)

    # ── POST ─────────────────────────────────────────────────────────────────

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        body = self._read_body()

        if path == "/api/predict":
            if not _READY:
                self._send_json({"error": "Model not ready"}, 503); return
            try:
                data   = json.loads(body.decode("utf-8"))
                result = ml.predict_patient(data, _STATE)
                self._send_json(result)
            except json.JSONDecodeError as exc:
                self._send_json({"error": f"Invalid JSON: {exc}"}, 400)
            except Exception as exc:
                import traceback
                self._send_json({"error": str(exc),
                                 "trace": traceback.format_exc()}, 500)
            return

        if path == "/api/batch":
            if not _READY:
                self._send_json({"error": "Model not ready"}, 503); return
            try:
                ct = self.headers.get("Content-Type", "")
                if "multipart" in ct:
                    boundary = ct.split("boundary=")[-1].encode()
                    parts    = body.split(b"--" + boundary)
                    csv_text = ""
                    for part in parts:
                        if b"filename=" in part:
                            idx = part.find(b"\r\n\r\n")
                            if idx >= 0:
                                csv_text = part[idx + 4:].rstrip(b"\r\n--").decode(
                                    "utf-8", errors="replace")
                                break
                else:
                    csv_text = body.decode("utf-8", errors="replace")

                if not csv_text.strip():
                    self._send_json({"error": "Empty CSV body"}, 400); return

                results = ml.batch_predict_csv(csv_text, _STATE)
                self._send_json({"results": results, "count": len(results)})
            except Exception as exc:
                import traceback
                self._send_json({"error": str(exc),
                                 "trace": traceback.format_exc()}, 500)
            return

        self._send_json({"error": f"Unknown route: {path}"}, 404)

    # ── File serving ─────────────────────────────────────────────────────────

    def _serve(self, filepath: str):
        if not os.path.isfile(filepath):
            self._send_json({"error": f"File not found: {filepath}"}, 404)
            return
        mime, _ = mimetypes.guess_type(filepath)
        mime    = mime or "application/octet-stream"
        with open(filepath, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type",   mime)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(port: int = 8000):
    print(f"\n{'='*58}")
    print("  VitalWatch — Patient Deterioration Prediction System")
    print(f"  Dashboard → http://localhost:{port}/")
    print(f"{'='*58}\n")

    t = threading.Thread(target=_train, daemon=True)
    t.start()
    print("[Server]  ML pipeline started in background thread.")
    print("[Server]  The dashboard will unlock once training completes (~30s).\n")

    httpd = HTTPServer(("0.0.0.0", port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server]  Shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    import os
    main(int(os.environ.get("PORT", 8000)))
