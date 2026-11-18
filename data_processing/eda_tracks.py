"""
Member 1 EDA: GPS track analysis.
Run from the repo root:  python -m data_processing.eda_tracks
Analysis uses the TRAIN partition; student_test is only counted (no routes used).
Output: results/eda_tracks.csv, results/eda_inventory.csv, results/eda_tracks_*.png
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_processing.load import load_trip, list_trips
from utils.geo import projected_distance_m

DATA_DIR = Path("data")
PARTITIONS = ["train", "student_test"]
JUMP_SPEED_MS = 50.0   # > 180 km/h between consecutive fixes = treat as GPS jump
BIG_GAP_S = 10.0       # time gap considered "large"
EXTS = ["track", "nodes", "arcs", "route"]


def inventory():
    """Trip counts, file presence and sizes per partition."""
    rows = []
    for part in PARTITIONS:
        for trip_dir in list_trips(DATA_DIR / part):
            tid = trip_dir.name
            row: dict[str, object] = {"partition": part, "trip_id": tid}
            for ext in EXTS:
                f = trip_dir / f"{tid}.{ext}"
                row[f"has_{ext}"] = f.exists()
                row[f"{ext}_kb"] = round(os.path.getsize(f) / 1024, 1) if f.exists() else None
            rows.append(row)
    return pd.DataFrame(rows)


def track_stats(track):
    """Quality and movement statistics for one GPS track."""
    lon, lat, t = track["lon"].to_numpy(), track["lat"].to_numpy(), track["t"].to_numpy()
    dt = np.diff(t)
    step_m = projected_distance_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step_m / dt, np.nan)

    return {
        "n_points": len(track),
        "n_nan": int(track.isna().sum().sum()),
        "n_bad_coords": int(((track["lon"].abs() > 180) | (track["lat"].abs() > 90)).sum()),
        "n_duplicate_rows": int(track.duplicated().sum()),
        "n_duplicate_t": int(track["t"].duplicated().sum()),
        "t_monotonic": bool(np.all(dt > 0)),
        "n_t_backwards": int((dt < 0).sum()),
        "duration_s": float(t[-1] - t[0]),
        "pct_dt_not_1s": round(float(np.mean(~np.isclose(dt, 1.0)) * 100), 2),
        "max_gap_s": float(dt.max()),
        "n_gaps_over_10s": int((dt > BIG_GAP_S).sum()),
        "length_km": round(float(step_m.sum() / 1000), 3),
        "speed_median_ms": round(float(np.nanmedian(speed)), 2),
        "speed_p95_ms": round(float(np.nanpercentile(speed, 95)), 2),
        "speed_max_ms": round(float(np.nanmax(speed)), 2),
        "n_jumps": int(np.nansum(speed > JUMP_SPEED_MS)),
        "lon_min": lon.min(), "lon_max": lon.max(),
        "lat_min": lat.min(), "lat_max": lat.max(),
    }, dt, speed


def main():
    os.makedirs("results", exist_ok=True)

    inv = inventory()
    inv.to_csv("results/eda_inventory.csv", index=False)
    print("Trips per partition:\n", inv.groupby("partition").size(), "\n")

    rows, all_dt, all_speed, all_pts = [], [], [], []
    for trip_dir in list_trips(DATA_DIR / "train"):
        track = load_trip(trip_dir, load_route=False)["track"]
        stats, dt, speed = track_stats(track)
        rows.append({"trip_id": trip_dir.name, **stats})
        all_dt.append(dt)
        all_speed.append(speed)
        all_pts.append(track[["lon", "lat"]])
        print(f"{trip_dir.name}: n={stats['n_points']} jumps={stats['n_jumps']} max_gap={stats['max_gap_s']}s")

    df = pd.DataFrame(rows)
    df.to_csv("results/eda_tracks.csv", index=False)
    print("\nSaved results/eda_tracks.csv")
    print(df.describe().T)

    dt_all = np.concatenate(all_dt)
    sp_all = np.concatenate(all_speed)
    pts = pd.concat(all_pts)

    # 1. Points per trip
    plt.figure(figsize=(7, 4))
    plt.hist(df["n_points"], bins=30)
    plt.xlabel("GPS points per trip"); plt.ylabel("Trips")
    plt.title("Trajectory length (train)")
    plt.tight_layout(); plt.savefig("results/eda_tracks_points_per_trip.png", dpi=150); plt.close()

    # 2. Sampling interval distribution
    plt.figure(figsize=(7, 4))
    vals, counts = np.unique(np.round(dt_all, 1), return_counts=True)
    top = np.argsort(counts)[::-1][:15]
    plt.bar([str(v) for v in vals[top]], counts[top])
    plt.yscale("log"); plt.xlabel("Time step (s)"); plt.ylabel("Count (log)")
    plt.title("Sampling intervals, 15 most common (train)")
    plt.tight_layout(); plt.savefig("results/eda_tracks_dt.png", dpi=150); plt.close()

    # 3. Speed distribution
    plt.figure(figsize=(7, 4))
    plt.hist(np.clip(sp_all[~np.isnan(sp_all)], 0, 80), bins=80)
    plt.axvline(JUMP_SPEED_MS, color="red", linestyle="--", label=f"jump threshold {JUMP_SPEED_MS} m/s")
    plt.xlabel("Speed between consecutive fixes (m/s, clipped at 80)"); plt.ylabel("Steps")
    plt.title("Implied speed (train)"); plt.legend()
    plt.tight_layout(); plt.savefig("results/eda_tracks_speed.png", dpi=150); plt.close()

    # 4. Coordinate distribution of all GPS points
    plt.figure(figsize=(6, 6))
    plt.scatter(pts["lon"], pts["lat"], s=0.2, alpha=0.3)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("All GPS observations (train)")
    plt.tight_layout(); plt.savefig("results/eda_tracks_coords.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()