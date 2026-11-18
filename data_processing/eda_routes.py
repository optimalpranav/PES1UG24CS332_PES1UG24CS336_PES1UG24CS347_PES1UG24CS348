"""
Member 4 EDA: reference route analysis.
Run from the repo root:  python -m data_processing.eda_routes
Uses TRAIN partition only.
Output: results/eda_routes.csv, results/eda_route_gps_dist.png, results/eda_route_breaks.png
"""
import numpy as np
import pandas as pd
import shapely
import matplotlib.pyplot as plt
from utils.geo import lonlat_to_xy as to_utm
from shapely.geometry import MultiLineString
from data_processing.load import load_trip, list_trips

TRAIN_DIR = "data/train"
OUT_CSV = "results/eda_routes.csv"

def route_checks(arcs, route_ids):
    """Validity, continuity and repetition of the reference arc sequence."""
    n_arcs = len(arcs)
    valid = (route_ids >= 0) & (route_ids < n_arcs)
    ids = route_ids[valid]

    frm = arcs["from_node"].to_numpy()[ids]
    to = arcs["to_node"].to_numpy()[ids]
    # a break = this arc's end node is not the next arc's start node
    breaks = int(np.sum(to[:-1] != frm[1:])) if len(ids) > 1 else 0
    consecutive_dups = int(np.sum(ids[:-1] == ids[1:])) if len(ids) > 1 else 0
    repeated = int(len(ids) - len(np.unique(ids)))

    return {
        "route_len": len(route_ids),
        "invalid_arc_ids": int((~valid).sum()),
        "unique_arcs": int(len(np.unique(ids))),
        "breaks": breaks,
        "consecutive_dup_arcs": consecutive_dups,
        "repeated_arcs": repeated,
    }, ids


def gps_to_route_distances(track, nodes, arcs, route_ids):
    """Distance (m) from every GPS point to the nearest arc on the reference route."""
    nx_, ny_ = to_utm(nodes["lon"], nodes["lat"])
    a = arcs.loc[route_ids]
    p0 = np.column_stack([nx_[a["from_node"]], ny_[a["from_node"]]])
    p1 = np.column_stack([nx_[a["to_node"]], ny_[a["to_node"]]])
    route_geom = MultiLineString([[tuple(s), tuple(e)] for s, e in zip(p0, p1)])
    tx, ty = to_utm(track["lon"], track["lat"])
    pts = shapely.points(tx, ty)
    return shapely.distance(pts, route_geom)


def main():
    rows, all_dists = [], []
    for trip_dir in list_trips(TRAIN_DIR):
        trip = load_trip(trip_dir)
        if trip["route"] is None:
            rows.append({"trip_id": trip["trip_id"], "has_route": False})
            continue

        route_ids = trip["route"]["arc_id"].to_numpy()
        stats, valid_ids = route_checks(trip["arcs"], route_ids)
        d = gps_to_route_distances(trip["track"], trip["nodes"], trip["arcs"], valid_ids)
        all_dists.append(d)

        rows.append({
            "trip_id": trip["trip_id"],
            "has_route": True,
            "n_gps": len(trip["track"]),
            **stats,
            "gps_dist_mean_m": round(float(d.mean()), 2),
            "gps_dist_median_m": round(float(np.median(d)), 2),
            "gps_dist_p95_m": round(float(np.percentile(d, 95)), 2),
            "gps_dist_max_m": round(float(d.max()), 2),
            "pct_gps_over_30m": round(float((d > 30).mean() * 100), 2),
        })
        print(f"{trip['trip_id']}: route_len={stats['route_len']} breaks={stats['breaks']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
    print(df.describe().T)

    # Plot 1: distribution of GPS-to-route distance (all train trips pooled)
    d_all = np.concatenate(all_dists)
    plt.figure(figsize=(7, 4))
    plt.hist(np.clip(d_all, 0, 100), bins=100)
    plt.xlabel("Distance from GPS point to reference route (m, clipped at 100)")
    plt.ylabel("GPS points")
    plt.title("GPS noise vs reference route (train partition)")
    plt.tight_layout()
    plt.savefig("results/eda_route_gps_dist.png", dpi=150)

    # Plot 2: route breaks per trip
    ok = df[df["has_route"]]
    plt.figure(figsize=(10, 4))
    plt.bar(ok["trip_id"], ok["breaks"])
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("Discontinuities in route")
    plt.title("Reference route breaks per trip (train partition)")
    plt.tight_layout()
    plt.savefig("results/eda_route_breaks.png", dpi=150)


if __name__ == "__main__":
    main()