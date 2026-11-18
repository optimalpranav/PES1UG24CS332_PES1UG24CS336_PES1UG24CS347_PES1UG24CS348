"""
Compare classifier-only vs classifier + HMM/Viterbi on a partition.

Run from repo root:
    python -m hmm.run_hmm                               # student_test, beta grid
    python -m hmm.run_hmm --betas 5 10 20 --partition student_test
Outputs: results/hmm_beta_search.csv, results/hmm_per_trip.csv,
         results/hmm_vs_classifier.csv
Decoder parameters are chosen on student_test only (allowed by the guidelines);
the evaluator-test partition is never used here.
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_processing.load import load_trip
from hmm.viterbi import RoadGraph, viterbi_trip
from utils.metrics import route_continuity, top1_predictions

MODEL_PATH = "models/saved/final_model.joblib"


def load_model(path=MODEL_PATH):
    obj = joblib.load(path)
    if isinstance(obj, dict):                 # optuna_tune.py saves a dict
        return obj["model"], obj["features"]
    from features.candidates import FEATURE_COLS
    return obj, FEATURE_COLS


def evaluate(cands, prob, data_dir, beta, emission_weight):
    """Per-trip accuracy and continuity for classifier-only and HMM."""
    cls_pred = top1_predictions(cands, prob)
    cands = cands.assign(_prob=prob)
    rows = []
    for tid, g in cands.groupby("trip_id", sort=True):
        trip = load_trip(Path(data_dir) / tid, load_route=False)
        graph = RoadGraph(trip["nodes"], trip["arcs"])
        truth = g.groupby("obs_idx")["true_arc"].first()

        t0 = time.perf_counter()
        hmm_map, breaks = viterbi_trip(g, g["_prob"].to_numpy(), trip["track"], graph,
                                       beta=beta, emission_weight=emission_weight)
        decode_s = time.perf_counter() - t0

        hmm_arcs = pd.Series(hmm_map).reindex(truth.index)
        cls_arcs = cls_pred[cls_pred.trip_id == tid].set_index("obs_idx")["pred_arc"].reindex(truth.index)

        rows.append({
            "trip_id": tid,
            "n_points": len(truth),
            "cls_top1_acc": float((cls_arcs == truth).mean()),
            "hmm_top1_acc": float((hmm_arcs == truth).mean()),
            "cls_continuity": route_continuity(cls_arcs.to_numpy(), trip["arcs"]),
            "hmm_continuity": route_continuity(hmm_arcs.to_numpy(), trip["arcs"]),
            "hmm_breaks": breaks,
            "decode_s": round(decode_s, 2),
        })
    return pd.DataFrame(rows)


def summarise(per_trip):
    w = per_trip["n_points"]
    return {
        "cls_top1_acc": float(np.average(per_trip["cls_top1_acc"], weights=w)),
        "hmm_top1_acc": float(np.average(per_trip["hmm_top1_acc"], weights=w)),
        "cls_continuity": float(per_trip["cls_continuity"].mean()),
        "hmm_continuity": float(per_trip["hmm_continuity"].mean()),
        "trips_improved": int((per_trip["hmm_top1_acc"] > per_trip["cls_top1_acc"]).sum()),
        "trips_worse": int((per_trip["hmm_top1_acc"] < per_trip["cls_top1_acc"]).sum()),
        "total_breaks": int(per_trip["hmm_breaks"].sum()),
        "decode_s_total": float(per_trip["decode_s"].sum()),
    }


def main(partition, betas, emission_weight):
    cands = pd.read_parquet(f"features/out/{partition}.parquet")
    model, feats = load_model()
    prob = model.predict_proba(cands[feats].to_numpy())[:, 1]
    data_dir = Path("data") / partition

    search, per_trip_by_beta = [], {}
    for beta in betas:
        per_trip = evaluate(cands, prob, data_dir, beta, emission_weight)
        s = {"beta": beta, "emission_weight": emission_weight, **summarise(per_trip)}
        search.append(s)
        per_trip_by_beta[beta] = per_trip
        print(f"beta={beta}: cls_top1={s['cls_top1_acc']:.4f} hmm_top1={s['hmm_top1_acc']:.4f} "
              f"cls_cont={s['cls_continuity']:.4f} hmm_cont={s['hmm_continuity']:.4f} "
              f"breaks={s['total_breaks']}")

    search = pd.DataFrame(search)
    search.to_csv("results/hmm_beta_search.csv", index=False)

    best = search.loc[search["hmm_top1_acc"].idxmax()]
    best_beta = best["beta"]
    per_trip_by_beta[best_beta].to_csv("results/hmm_per_trip.csv", index=False)
    pd.DataFrame([
        {"method": "classifier only", "top1_acc": best["cls_top1_acc"], "route_continuity": best["cls_continuity"]},
        {"method": f"classifier + HMM (beta={best_beta})", "top1_acc": best["hmm_top1_acc"],
         "route_continuity": best["hmm_continuity"]},
    ]).to_csv("results/hmm_vs_classifier.csv", index=False)
    print(f"\nBest beta = {best_beta}. Saved results/hmm_*.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--betas", type=float, nargs="+", default=[2, 5, 10, 20, 50])
    ap.add_argument("--emission-weight", type=float, default=1.0)
    a = ap.parse_args()
    main(a.partition, a.betas, a.emission_weight)