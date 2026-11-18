"""Shared metrics (Members 2 and 3)."""
import numpy as np
import pandas as pd


def top1_predictions(df, prob):
    """Per GPS point, the candidate arc with the highest score."""
    d = df[["trip_id", "obs_idx", "arc_id"]].copy()
    d["prob"] = prob
    best = d.loc[d.groupby(["trip_id", "obs_idx"])["prob"].idxmax()]
    return best[["trip_id", "obs_idx", "arc_id"]].rename(columns={"arc_id": "pred_arc"})


def top1_accuracy(df, pred):
    """Observation-level Top-1 accuracy over points that have candidates.
    pred: DataFrame trip_id, obs_idx, pred_arc."""
    truth = df.groupby(["trip_id", "obs_idx"])["true_arc"].first().reset_index()
    m = truth.merge(pred, on=["trip_id", "obs_idx"], how="left")
    return float((m["pred_arc"] == m["true_arc"]).mean())


def route_continuity(pred_arcs, arcs):
    """Share of consecutive changes of predicted arc where the two arcs connect
    (to_node of the first == from_node of the second)."""
    p = np.asarray([a for a in pred_arcs if a is not None and a >= 0])
    if len(p) < 2:
        return float("nan")
    a, b = p[:-1], p[1:]
    change = a != b
    if not change.any():
        return 1.0
    to_n = arcs["to_node"].to_numpy()[a[change]]
    from_n = arcs["from_node"].to_numpy()[b[change]]
    return float((to_n == from_n).mean())