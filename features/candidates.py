"""
Member 1: candidate generation, per-point labels and features.

For every GPS point we take the K nearest road arcs within RADIUS_M metres
(exact point-to-segment distance in UTM 36N metres) and compute features for
each (point, arc) pair. One row = one candidate.

Labels (train/student_test only, never used as features):
the .route file is an ordered list of arcs, not one arc per GPS point, so we
align points to route arcs with a monotonic DP: point i is assigned route
position j(i), j never decreases along the track, and the summed
point-to-arc distance is minimal. label = 1 if the candidate arc is that arc.

Run from repo root:
    python -m features.candidates            # builds train + student_test
Outputs: features/out/<partition>.parquet, results/folds.csv,
         results/candidate_miss_rate.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_processing.load import load_trip, list_trips
from utils.geo import lonlat_to_xy

K = 8              # max candidates per GPS point
RADIUS_M = 60.0    # max point-to-arc distance
OUT_DIR = Path("features/out")
N_FOLDS = 5
SEED = 42

FEATURE_COLS = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min",
    "heading_diff", "speed", "arc_len", "proj_frac",
    "n_cands", "dt_prev", "dist_prev_change",
]


# ---------- geometry ----------
def arc_geometry(nodes, arcs):
    """Projected start/end coordinates, length and bearing of every arc."""
    nx_, ny_ = lonlat_to_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    f, t = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    ax, ay, bx, by = nx_[f], ny_[f], nx_[t], ny_[t]
    length = np.hypot(bx - ax, by - ay)
    bearing = np.degrees(np.arctan2(bx - ax, by - ay)) % 360
    return ax, ay, bx, by, length, bearing


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments and the fraction along the segment."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    frac = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.maximum(l2, 1e-9), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    qx, qy = ax + frac * dx, ay + frac * dy
    return np.hypot(px - qx, py - qy), frac


def gps_kinematics(px, py, t):
    """Heading (deg from north) and speed (m/s) from neighbouring points."""
    n = len(px)
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    prv = np.maximum(np.arange(n) - 1, 0)
    hx, hy = px[nxt] - px[prv], py[nxt] - py[prv]
    heading = np.degrees(np.arctan2(hx, hy)) % 360
    moving = np.hypot(hx, hy) > 1.0          # heading is meaningless when stopped
    dt = np.maximum(t[nxt] - t[prv], 1e-6)
    speed = np.hypot(hx, hy) / dt
    dt_prev = np.r_[0.0, np.diff(t)]
    return heading, moving, speed, dt_prev


# ---------- labels ----------
def align_route(px, py, route_arcs, geom):
    """Monotonic DP alignment of GPS points to the ordered route arcs.
    Returns the arc_id assigned to each point."""
    ax, ay, bx, by, _, _ = geom
    r = np.asarray(route_arcs)
    d, _ = point_segment(px[:, None], py[:, None], ax[r][None], ay[r][None], bx[r][None], by[r][None])
    n, m = d.shape
    cost = d[0].copy()
    back = np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        # best previous position j' <= j: running minimum over the row
        best = np.minimum.accumulate(cost)
        idx = np.maximum.accumulate(np.where(cost == best, np.arange(m), 0))
        back[i] = idx
        cost = best + d[i]
    j = int(np.argmin(cost))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    return r[path]


# ---------- candidates + features ----------
def build_candidates(trip, with_labels=True):
    """Candidate table for one trip. Features use GPS + network only;
    the route is used only for the label column."""
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    track = track.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    px, py = lonlat_to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    t = track["t"].to_numpy()
    geom = arc_geometry(nodes, arcs)
    ax, ay, bx, by, length, bearing = geom

    # KD-tree on arc midpoints; search radius covers half the longest arc
    mx, my = (ax + bx) / 2, (ay + by) / 2
    tree = cKDTree(np.c_[mx, my])
    search_r = RADIUS_M + length.max() / 2
    heading, moving, speed, dt_prev = gps_kinematics(px, py, t)

    rows = []
    for i, near in enumerate(tree.query_ball_point(np.c_[px, py], r=search_r)):
        if not near:
            continue
        near = np.asarray(near)
        d, frac = point_segment(px[i], py[i], ax[near], ay[near], bx[near], by[near])
        keep = d <= RADIUS_M
        if not keep.any():
            continue
        near, d, frac = near[keep], d[keep], frac[keep]
        order = np.argsort(d)[:K]
        near, d, frac = near[order], d[order], frac[order]
        hd = np.abs((heading[i] - bearing[near] + 180) % 360 - 180)
        if not moving[i]:
            hd = np.full_like(hd, 90.0)   # neutral value when stopped
        for rank, (a, di, fi, hdi) in enumerate(zip(near, d, frac, hd)):
            rows.append((i, t[i], int(a), di, rank, di - d[0], di / (d[0] + 1.0),
                         hdi, speed[i], length[a], fi, len(near), dt_prev[i]))

    df = pd.DataFrame(rows, columns=["obs_idx", "timestamp", "arc_id", "dist", "dist_rank",
                                     "dist_minus_min", "dist_ratio_min", "heading_diff",
                                     "speed", "arc_len", "proj_frac", "n_cands", "dt_prev"])
    df.insert(0, "trip_id", trip["trip_id"])

    # change in distance to the same arc compared with the previous point
    prev = df[["obs_idx", "arc_id", "dist"]].copy()
    prev["obs_idx"] += 1
    df = df.merge(prev.rename(columns={"dist": "dist_prev"}), on=["obs_idx", "arc_id"], how="left")
    df["dist_prev_change"] = (df["dist"] - df["dist_prev"]).fillna(0.0)
    df = df.drop(columns="dist_prev")

    if with_labels and trip["route"] is not None:
        true_arc = align_route(px, py, trip["route"]["arc_id"].to_numpy(), geom)
        df["true_arc"] = true_arc[df["obs_idx"].to_numpy()]
        df["label"] = (df["arc_id"] == df["true_arc"]).astype(int)
    return df.sort_values(["obs_idx", "dist_rank"]).reset_index(drop=True)


def true_arcs_for_trip(trip):
    """Per-point reference arc (used server-side by the API for accuracy)."""
    track = trip["track"].drop_duplicates("t").sort_values("t").reset_index(drop=True)
    px, py = lonlat_to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    geom = arc_geometry(trip["nodes"], trip["arcs"])
    return align_route(px, py, trip["route"]["arc_id"].to_numpy(), geom)


def build_partition(partition):
    tables, miss = [], []
    for trip_dir in list_trips(Path("data") / partition):
        trip = load_trip(trip_dir)
        df = build_candidates(trip)
        n_obs = trip["track"]["t"].nunique()
        hit = df.groupby("obs_idx")["label"].max()
        miss.append((trip["trip_id"], n_obs, len(hit), 1 - hit.sum() / n_obs))
        tables.append(df)
        print(f"{partition} {trip['trip_id']}: {len(df)} rows, miss rate {miss[-1][3]:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_parquet(OUT_DIR / f"{partition}.parquet", index=False)
    return pd.DataFrame(miss, columns=["trip_id", "n_obs", "n_obs_with_cands", "candidate_miss_rate"])


def make_folds(train_ids):
    rng = np.random.default_rng(SEED)
    ids = np.array(sorted(train_ids))
    rng.shuffle(ids)
    return pd.DataFrame({"trip_id": ids, "fold": np.arange(len(ids)) % N_FOLDS})


if __name__ == "__main__":
    parts = sys.argv[1:] or ["train", "student_test"]
    Path("results").mkdir(exist_ok=True)
    all_miss = []
    for p in parts:
        m = build_partition(p)
        m.insert(0, "partition", p)
        all_miss.append(m)
    miss = pd.concat(all_miss)
    miss.to_csv("results/candidate_miss_rate.csv", index=False)
    print(miss.groupby("partition")["candidate_miss_rate"].mean())
    if "train" in parts:
        make_folds(miss.loc[miss.partition == "train", "trip_id"]).to_csv("results/folds.csv", index=False)
"""
Member 1: candidate generation, per-point labels and features.

For every GPS point we take the K nearest road arcs within RADIUS_M metres
(exact point-to-segment distance in UTM 36N metres) and compute features for
each (point, arc) pair. One row = one candidate.

Labels (train/student_test only, never used as features):
the .route file is an ordered list of arcs, not one arc per GPS point, so we
align points to route arcs with a monotonic DP: point i is assigned route
position j(i), j never decreases along the track, and the summed
point-to-arc distance is minimal. label = 1 if the candidate arc is that arc.

Run from repo root:
    python -m features.candidates            # builds train + student_test
Outputs: features/out/<partition>.parquet, results/folds.csv,
         results/candidate_miss_rate.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_processing.load import load_trip, list_trips
from utils.geo import lonlat_to_xy

K = 8              # max candidates per GPS point
RADIUS_M = 60.0    # max point-to-arc distance
OUT_DIR = Path("features/out")
N_FOLDS = 5
SEED = 42

FEATURE_COLS = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min",
    "heading_diff", "speed", "arc_len", "proj_frac",
    "n_cands", "dt_prev", "dist_prev_change",
]


# ---------- geometry ----------
def arc_geometry(nodes, arcs):
    """Projected start/end coordinates, length and bearing of every arc."""
    nx_, ny_ = lonlat_to_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    f, t = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    ax, ay, bx, by = nx_[f], ny_[f], nx_[t], ny_[t]
    length = np.hypot(bx - ax, by - ay)
    bearing = np.degrees(np.arctan2(bx - ax, by - ay)) % 360
    return ax, ay, bx, by, length, bearing


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments and the fraction along the segment."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    frac = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.maximum(l2, 1e-9), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    qx, qy = ax + frac * dx, ay + frac * dy
    return np.hypot(px - qx, py - qy), frac


def gps_kinematics(px, py, t):
    """Heading (deg from north) and speed (m/s) from neighbouring points."""
    n = len(px)
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    prv = np.maximum(np.arange(n) - 1, 0)
    hx, hy = px[nxt] - px[prv], py[nxt] - py[prv]
    heading = np.degrees(np.arctan2(hx, hy)) % 360
    moving = np.hypot(hx, hy) > 1.0          # heading is meaningless when stopped
    dt = np.maximum(t[nxt] - t[prv], 1e-6)
    speed = np.hypot(hx, hy) / dt
    dt_prev = np.r_[0.0, np.diff(t)]
    return heading, moving, speed, dt_prev


# ---------- labels ----------
def align_route(px, py, route_arcs, geom):
    """Monotonic DP alignment of GPS points to the ordered route arcs.
    Returns the arc_id assigned to each point."""
    ax, ay, bx, by, _, _ = geom
    r = np.asarray(route_arcs)
    d, _ = point_segment(px[:, None], py[:, None], ax[r][None], ay[r][None], bx[r][None], by[r][None])
    n, m = d.shape
    cost = d[0].copy()
    back = np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        # best previous position j' <= j: running minimum over the row
        best = np.minimum.accumulate(cost)
        idx = np.maximum.accumulate(np.where(cost == best, np.arange(m), 0))
        back[i] = idx
        cost = best + d[i]
    j = int(np.argmin(cost))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    return r[path]


# ---------- candidates + features ----------
def build_candidates(trip, with_labels=True):
    """Candidate table for one trip. Features use GPS + network only;
    the route is used only for the label column."""
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    track = track.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    px, py = lonlat_to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    t = track["t"].to_numpy()
    geom = arc_geometry(nodes, arcs)
    ax, ay, bx, by, length, bearing = geom

    # KD-tree on arc midpoints; search radius covers half the longest arc
    mx, my = (ax + bx) / 2, (ay + by) / 2
    tree = cKDTree(np.c_[mx, my])
    search_r = RADIUS_M + length.max() / 2
    heading, moving, speed, dt_prev = gps_kinematics(px, py, t)

    rows = []
    for i, near in enumerate(tree.query_ball_point(np.c_[px, py], r=search_r)):
        if not near:
            continue
        near = np.asarray(near)
        d, frac = point_segment(px[i], py[i], ax[near], ay[near], bx[near], by[near])
        keep = d <= RADIUS_M
        if not keep.any():
            continue
        near, d, frac = near[keep], d[keep], frac[keep]
        order = np.argsort(d)[:K]
        near, d, frac = near[order], d[order], frac[order]
        hd = np.abs((heading[i] - bearing[near] + 180) % 360 - 180)
        if not moving[i]:
            hd = np.full_like(hd, 90.0)   # neutral value when stopped
        for rank, (a, di, fi, hdi) in enumerate(zip(near, d, frac, hd)):
            rows.append((i, t[i], int(a), di, rank, di - d[0], di / (d[0] + 1.0),
                         hdi, speed[i], length[a], fi, len(near), dt_prev[i]))

    df = pd.DataFrame(rows, columns=["obs_idx", "timestamp", "arc_id", "dist", "dist_rank",
                                     "dist_minus_min", "dist_ratio_min", "heading_diff",
                                     "speed", "arc_len", "proj_frac", "n_cands", "dt_prev"])
    df.insert(0, "trip_id", trip["trip_id"])

    # change in distance to the same arc compared with the previous point
    prev = df[["obs_idx", "arc_id", "dist"]].copy()
    prev["obs_idx"] += 1
    df = df.merge(prev.rename(columns={"dist": "dist_prev"}), on=["obs_idx", "arc_id"], how="left")
    df["dist_prev_change"] = (df["dist"] - df["dist_prev"]).fillna(0.0)
    df = df.drop(columns="dist_prev")

    if with_labels and trip["route"] is not None:
        true_arc = align_route(px, py, trip["route"]["arc_id"].to_numpy(), geom)
        df["true_arc"] = true_arc[df["obs_idx"].to_numpy()]
        df["label"] = (df["arc_id"] == df["true_arc"]).astype(int)
    return df.sort_values(["obs_idx", "dist_rank"]).reset_index(drop=True)


def true_arcs_for_trip(trip):
    """Per-point reference arc (used server-side by the API for accuracy)."""
    track = trip["track"].drop_duplicates("t").sort_values("t").reset_index(drop=True)
    px, py = lonlat_to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    geom = arc_geometry(trip["nodes"], trip["arcs"])
    return align_route(px, py, trip["route"]["arc_id"].to_numpy(), geom)


def build_partition(partition):
    tables, miss = [], []
    
        trip = load_trip(trip_dir)
        df = build_candidates(trip)
        n_obs = trip["track"]["t"].nunique()
        hit = df.groupby("obs_idx")["label"].max()
        miss.append((trip["trip_id"], n_obs, len(hit), 1 - hit.sum() / n_obs))
        tables.append(df)
        print(f"{partition} {trip['trip_id']}: {len(df)} rows, miss rate {miss[-1][3]:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_parquet(OUT_DIR / f"{partition}.parquet", index=False)
    return pd.DataFrame(miss, columns=["trip_id", "n_obs", "n_obs_with_cands", "candidate_miss_rate"])


def make_folds(train_ids):
    rng = np.random.default_rng(SEED)
    ids = np.array(sorted(train_ids))
    rng.shuffle(ids)
    return pd.DataFrame({"trip_id": ids, "fold": np.arange(len(ids)) % N_FOLDS})


if __name__ == "__main__":
    parts = sys.argv[1:] or ["train", "student_test"]
    Path("results").mkdir(exist_ok=True)
    all_miss = []
    for p in parts:
        m = build_partition(p)
        m.insert(0, "partition", p)
        all_miss.append(m)
    miss = pd.concat(all_miss)
    miss.to_csv("results/candidate_miss_rate.csv", index=False)
    print(miss.groupby("partition")["candidate_miss_rate"].mean())
    if "train" in parts:
        make_folds(miss.loc[miss.partition == "train", "trip_id"]).to_csv("results/folds.csv", index=False)