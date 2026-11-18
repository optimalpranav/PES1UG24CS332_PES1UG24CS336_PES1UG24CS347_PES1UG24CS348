"""
PathSnap API (Laptop 1).
Run from repo root:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.config import EVAL_DIR
from api.decoder import viterbi_decode
from api.model_loader import load_model
from api.pipeline import list_eval_trips, predict_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")
STATE = {}


@asynccontextmanager
async def lifespan(app):
    STATE["bundle"] = load_model()
    b = STATE["bundle"]
    log.info("Model: %s (%s)  params=%s", b["name"], b["path"], b["params"])
    log.info("Decoder: %s", "viterbi" if viterbi_decode else "argmax (classifier-only)")
    log.info("Eval trips (%s): %s", EVAL_DIR, list_eval_trips())
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s from %s -> %d in %.0f ms", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code,
             (time.perf_counter() - t0) * 1000)
    return response


@app.get("/health")
def health():
    b = STATE["bundle"]
    return {"status": "ok", "model": b["name"], "decoder": "viterbi" if viterbi_decode else "argmax"}


@app.get("/trips")
def trips():
    return {"trips": [{"trip_id": i} for i in list_eval_trips()]}


@app.post("/predict")
def predict(req: PredictRequest):   # sync def -> runs in threadpool, doesn't block the server
    if req.trip_id not in list_eval_trips():   # also blocks path tricks like "../train/x"
        raise HTTPException(404, f"Unknown trip_id {req.trip_id}")
    log.info("Predict request for trip %s", req.trip_id)
    try:
        res = predict_trip(req.trip_id, STATE["bundle"])
    except ValueError as e:
        raise HTTPException(422, str(e))
    m = res["metrics"]
    log.info("Trip %s | decoder=%s | top1=%.4f | continuity=%.4f | %d pts | %.0f ms",
             req.trip_id, res["decoder"], m.get("top1_acc", float("nan")),
             m.get("route_continuity", float("nan")), m["n_points"], res["timing_ms"]["total"])
    return res