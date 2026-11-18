"""Geodesy helpers for the trajectory (trip) data.

Provides lon/lat -> UTM projection (EPSG:32636 = UTM zone 36N) via pyproj,
plus a projected (Euclidean) distance function and a haversine great-circle
reference so the projection choice can be justified in the report.

Notes
-----
EPSG:32636 covers longitude 18E-24E, northern hemisphere (East Africa: Kenya,
Tanzania, Uganda). The train trips in this project are worldwide (Germany,
Russia, Argentina, Japan, ...), so a single fixed UTM zone is only accurate
for points that fall inside zone 36N. For production use across all trips,
pick the UTM zone per trip/point. The fixed EPSG:32636 implementation below is
kept because it is what was specified and what Members 2 & 4 consume early.
"""

from __future__ import annotations

import math
from typing import Tuple

import pyproj

# UTM zone 36N, WGS84 datum -> coordinates in metres.
UTM36N = "EPSG:32636"
WGS84 = "EPSG:4326"

# Transformer: (lon, lat) in degrees -> (easting, northing) in metres.
_to_utm36n = pyproj.Transformer.from_crs(WGS84, UTM36N, always_xy=True)
_to_wgs84 = pyproj.Transformer.from_crs(UTM36N, WGS84, always_xy=True)


def lonlat_to_utm36n(lon: float, lat: float) -> Tuple[float, float]:
    """Project (lon, lat) [degrees] to UTM 36N (easting, northing) [metres]."""
    easting, northing = _to_utm36n.transform(lon, lat)
    return float(easting), float(northing)


def utm36n_to_lonlat(easting: float, northing: float) -> Tuple[float, float]:
    """Inverse: UTM 36N (easting, northing) [metres] -> (lon, lat) [degrees]."""
    lon, lat = _to_wgs84.transform(easting, northing)
    return float(lon), float(lat)


def distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Projected (flat-earth) distance between two points in metres.

    Both points are projected to UTM 36N and the Euclidean distance is taken.
    Only meaningful when both points lie inside (or near) UTM zone 36N.
    """
    e1, n1 = lonlat_to_utm36n(lon1, lat1)
    e2, n2 = lonlat_to_utm36n(lon2, lat2)
    return math.hypot(e2 - e1, n2 - n1)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle (haversine) distance between two points in metres.

    Reference against which the UTM-projected distance is compared.
    """
    R = 6_371_000.0  # mean Earth radius, metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compare_distances(lon1: float, lat1: float, lon2: float, lat2: float) -> dict:
    """Return projected vs haversine distance (metres) and their difference."""
    d_proj = distance_m(lon1, lat1, lon2, lat2)
    d_hav = haversine_m(lat1, lon1, lat2, lon2)
    return {
        "projected_m": d_proj,
        "haversine_m": d_hav,
        "abs_diff_m": abs(d_proj - d_hav),
        "rel_diff_percent": 100.0 * abs(d_proj - d_hav) / d_hav if d_hav else 0.0,
    }


if __name__ == "__main__":
    # Sanity check: two points inside UTM 36N (Nairobi, Kenya).
    pairs = [
        ("Nairobi pair", 36.8219, -1.2921, 36.8219, -1.2821),   # ~1.1 km north
        ("Nairobi pair 2", 36.8000, -1.2900, 36.8400, -1.2900), # ~4 km east
    ]
    for name, lon1, lat1, lon2, lat2 in pairs:
        r = compare_distances(lon1, lat1, lon2, lat2)
        print(f"{name}: projected={r['projected_m']:.2f} m  "
              f"haversine={r['haversine_m']:.2f} m  "
              f"diff={r['abs_diff_m']:.2f} m ({r['rel_diff_percent']:.4f}%)")
