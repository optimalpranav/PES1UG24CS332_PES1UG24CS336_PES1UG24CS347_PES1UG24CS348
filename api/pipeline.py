import time

import pandas as pd

from api.config import EVAL_DIR, NETWORK_MARGIN_DEG
from api.decoder import argmax_decode, decode
from data_processing.load import load_trip
from features.candidates import build_candidates, true_arcs_for_trip
from utils.metrics import route_continuity


def list_eval_trips():
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir())


def _segments(nodes, arcs, arc_ids):
    """[[[lat, lon], [lat, lon]], ...] for Folium (lat first)."""
    a = arcs.loc[list(arc_ids)]
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    return [[[float(lat[f]), float(lon[f])], [float(lat[t]), float(lon[t])]]
            for f, t in zip(a["from_node"], a["to_node"])]


def _network_near(track, nodes, arcs):
    la_min, la_max = track["lat"].min() - NETWORK_MARGIN_DEG, track["lat"].max() + NETWORK_MARGIN_DEG
    lo_min, lo_max = track["lon"].min() - NETWORK_MARGIN_DEG, track["lon"].max() + NETWORK_MARGIN_DEG
    inside = (nodes["lon"].between(lo_min, lo_max) & nodes["lat"].between(la_min, la_max)).to_numpy()
    keep = inside[arcs["from_node"]] | inside[arcs["to_node"]]
    return arcs.index[keep]


def _collapse(arc_seq):
    """Per-point arcs -> ordered route (drop consecutive repeats)."""
    out = []
    for a in arc_seq:
        if not out or out[-1] != a:
            out.append(int(a))
    return out


def _score(trip_id, trip, pred):
    """SERVER-SIDE ONLY: reads the reference route, returns numbers, never the route itself."""
    ref = load_trip(EVAL_DIR / trip_id)["route"]
    if ref is None:
        return None
    true_arc = pd.Series(true_arcs_for_trip({**trip, "route": ref}))   # index = obs_idx
    acc = float((pred == true_arc.reindex(pred.index)).mean())
    cont = route_continuity(pred.to_numpy(), trip["arcs"])
    return {"top1_acc": acc, "route_continuity": cont}


def predict_trip(trip_id, bundle):
    t = {"start": time.perf_counter()}

    # 1. prediction side: no route loaded at all
    trip = load_trip(EVAL_DIR / trip_id, load_route=False)
    t["load"] = time.perf_counter()

    cands = build_candidates(trip, with_labels=False)
    if cands.empty:
        raise ValueError(f"Trip {trip_id}: no road candidates within radius for any GPS point")
    t["features"] = time.perf_counter()

    cands["prob"] = bundle["model"].predict_proba(cands[bundle["features"]].to_numpy())[:, 1]
    t["classifier"] = time.perf_counter()

    pred, decoder = decode(cands, trip)
    t["decode"] = time.perf_counter()

    # 2. scoring side
    track = trip["track"].drop_duplicates("t").sort_values("t").reset_index(drop=True)
    metrics = {"n_points": int(len(track)), "points_with_candidates": int(pred.size)}
    final = _score(trip_id, trip, pred)
    if final:
        metrics.update(final)
        if decoder != "argmax":   # report the HMM gain, needed for the report anyway
            metrics["classifier_only"] = _score(trip_id, trip, argmax_decode(cands))
    t["score"] = time.perf_counter()

    steps = ["load", "features", "classifier", "decode", "score"]
    prev, timing = t["start"], {}
    for s in steps:
        timing[s] = round((t[s] - prev) * 1000, 1)
        prev = t[s]
    timing["total"] = round((t["score"] - t["start"]) * 1000, 1)

    return {
        "trip_id": trip_id,
        "model": bundle["name"],
        "decoder": decoder,
        "gps": track[["lat", "lon"]].to_numpy().tolist(),
        "network": _segments(trip["nodes"], trip["arcs"], _network_near(track, trip["nodes"], trip["arcs"])),
        "predicted_route": _segments(trip["nodes"], trip["arcs"], _collapse(pred.to_numpy())),
        "metrics": metrics,
        "timing_ms": timing,
    }