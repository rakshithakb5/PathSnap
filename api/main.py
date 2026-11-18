"""PHASE 6 - FastAPI backend (runs on LAPTOP 1).

Start (from the repo root):
    set PATHSNAP_EVAL_DIR=C:\\path\\to\\evaluator_test        (Windows CMD)   # folder with the 10 trip folders
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Contract with the UI (LAPTOP 2):
    GET  /health            -> {"status":"ok", ...}
    GET  /trips             -> {"trips": ["00000003", ...]}            the evaluation trips available on the server
    POST /predict {"trip_id": "00000003"}
         -> predicted route + GPS + evaluation metrics (computed SERVER-SIDE)

Leakage rules enforced in code:
  * the prediction pipeline gets a trip loaded with with_route=False  -> it physically cannot see the labels
  * the reference route is read only inside evaluate_prediction() AFTER the prediction exists
  * the response contains predictions and metrics, never the reference route / labels
Every request is logged to the console (this is what you show the evaluator as the 'API logs').
"""
import os, sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathsnap.config import DATA_DIR
from pathsnap.io import load_trip
from pathsnap.pipeline import PathSnapPipeline
from pathsnap.evaluation import evaluate_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", DATA_DIR / "student_test"))   # default: stand-in for local testing
app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
PIPE = None


@app.on_event("startup")
def _startup():
    global PIPE
    PIPE = PathSnapPipeline.load()
    log.info("Evaluation trips directory: %s  (%d trips)", EVAL_DIR, len(_trip_ids()))


def _trip_ids():
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir()) if EVAL_DIR.exists() else []


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": PIPE is not None, "eval_dir": str(EVAL_DIR), "n_trips": len(_trip_ids())}


@app.get("/trips")
def trips():
    return {"trips": _trip_ids()}


@app.post("/predict")
def predict(req: PredictRequest):
    t_start = time.perf_counter()
    tid = req.trip_id.strip()
    if tid not in _trip_ids():
        log.warning("REQUEST trip_id=%s -> 404 (not in evaluation set)", tid)
        raise HTTPException(404, f"unknown trip_id {tid}")
    log.info("REQUEST  POST /predict trip_id=%s", tid)
    d = EVAL_DIR / tid
    trip = load_trip(d, with_route=False)                          # NO ground truth reaches the pipeline
    res = PIPE.run(trip)
    hmm = res["hmm_pred"]
    # ---- server-side evaluation (the only place ground truth is read) ----
    metrics = None
    if (d / f"{tid}.route").exists():
        ref = load_trip(d, with_route=True)
        metrics = evaluate_prediction(ref, {"classifier_only": res["clf_pred"], "classifier_plus_hmm": hmm})
    # ---- response: GPS + predicted route geometry (lon/lat) ----
    route_arcs = [int(x) for x in dict.fromkeys(int(v) for v in hmm if v >= 0)]
    geom = [[[float(trip.nodes[trip.arcs[a, 0], 0]), float(trip.nodes[trip.arcs[a, 0], 1])],
             [float(trip.nodes[trip.arcs[a, 1], 0]), float(trip.nodes[trip.arcs[a, 1], 1])]] for a in route_arcs]
    total = time.perf_counter() - t_start
    out = dict(trip_id=tid, n_points=int(len(trip.xy)), gps=trip.track[:, :2].round(6).tolist(),
               predicted_arc_per_point=[int(v) for v in hmm], predicted_route_arcs=route_arcs,
               predicted_route_geometry=geom, metrics=metrics,
               timings={**{k: round(v, 3) for k, v in res["timings"].items()}, "total_s": round(total, 3)})
    if metrics:
        log.info("RESPONSE trip_id=%s points=%d | Top-1 clf-only=%.4f -> clf+HMM=%.4f | continuity %.3f -> %.3f | route F1 %.3f | %.2fs",
                 tid, out["n_points"], metrics["classifier_only"]["top1_accuracy"], metrics["classifier_plus_hmm"]["top1_accuracy"],
                 metrics["classifier_only"]["route_continuity"], metrics["classifier_plus_hmm"]["route_continuity"],
                 metrics["classifier_plus_hmm"]["route_f1"], total)
    else:
        log.info("RESPONSE trip_id=%s points=%d (no reference available) %.2fs", tid, out["n_points"], total)
    return out
