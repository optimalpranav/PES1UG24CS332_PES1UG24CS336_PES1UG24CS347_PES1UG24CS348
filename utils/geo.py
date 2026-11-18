"""Geodesy helpers for PathSnap.

All functions take (lon, lat) in that order, in degrees, and work on single
values or numpy arrays.

Projection: UTM zone 36N (EPSG:32636), units in metres.
UTM zone 36N spans longitudes 30E-36E in the northern hemisphere. The trips
are projected into it so that point-to-road distances, speeds and arc lengths
can be computed with plain Euclidean geometry in metres.
(Justify against the dataset: see the bounding-box check in __main__.)
"""

import numpy as np
from pyproj import Transformer

WGS84 = "EPSG:4326"
UTM36N = "EPSG:32636"
EARTH_RADIUS_M = 6_371_000.0

_to_utm = Transformer.from_crs(WGS84, UTM36N, always_xy=True)
_to_wgs = Transformer.from_crs(UTM36N, WGS84, always_xy=True)


def lonlat_to_xy(lon, lat):
    """(lon, lat) degrees -> (x, y) metres in UTM 36N. Scalars or arrays."""
    x, y = _to_utm.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return x, y


def xy_to_lonlat(x, y):
    """Inverse of lonlat_to_xy."""
    lon, lat = _to_wgs.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return lon, lat


def projected_distance_m(lon1, lat1, lon2, lat2):
    """Euclidean distance in metres after projecting both points to UTM 36N."""
    x1, y1 = lonlat_to_xy(lon1, lat1)
    x2, y2 = lonlat_to_xy(lon2, lat2)
    return np.hypot(x2 - x1, y2 - y1)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (reference for checking the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(lon1, lat1, lon2, lat2):
    """Direction of travel from point 1 to point 2, in degrees from north (0-360),
    measured on the projected plane. Useful for GPS heading and arc direction."""
    x1, y1 = lonlat_to_xy(lon1, lat1)
    x2, y2 = lonlat_to_xy(lon2, lat2)
    return np.degrees(np.arctan2(x2 - x1, y2 - y1)) % 360


if __name__ == "__main__":
    # Check the projection on our own data: run from repo root
    #   python -m utils.geo
    from data_processing.load import load_trip, list_trips

    trips = list_trips("data/train")
    rows = []
    for trip_dir in trips:
        tr = load_trip(trip_dir, load_route=False)["track"]
        lon, lat = tr["lon"].to_numpy(), tr["lat"].to_numpy()
        proj = projected_distance_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
        hav = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
        mask = hav > 1  # skip near-zero steps
        rel = np.abs(proj[mask] - hav[mask]) / hav[mask] * 100
        rows.append((trip_dir.name, lon.min(), lon.max(), lat.min(), lat.max(),
                     float(np.mean(rel)) if mask.any() else float("nan")))

    print(f"{'trip':10s} {'lon_min':>9s} {'lon_max':>9s} {'lat_min':>9s} {'lat_max':>9s} {'mean_rel_diff_%':>16s}")
    for r in rows:
        print(f"{r[0]:10s} {r[1]:9.4f} {r[2]:9.4f} {r[3]:9.4f} {r[4]:9.4f} {r[5]:16.4f}")
