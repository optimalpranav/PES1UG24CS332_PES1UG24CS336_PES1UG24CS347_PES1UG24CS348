"""HTTP calls to the PathSnap API (Laptop 1). The UI gets everything through here."""
import os

import requests

DEFAULT_API_URL = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")
PREDICT_TIMEOUT_S = 180


class ApiError(Exception):
    pass


def _call(method, base, path, timeout, **kw):
    url = base.rstrip("/") + path
    try:
        r = requests.request(method, url, timeout=timeout, **kw)
    except requests.RequestException as e:
        raise ApiError(f"Cannot reach API at {base}: {e}") from e
    if r.status_code != 200:
        raise ApiError(f"{method} {path} failed ({r.status_code}): {r.text[:300]}")
    return r.json()


def health(base):
    return _call("GET", base, "/health", timeout=5)


def list_trips(base):
    return [t["trip_id"] for t in _call("GET", base, "/trips", timeout=10)["trips"]]


def predict(base, trip_id):
    return _call("POST", base, "/predict", timeout=PREDICT_TIMEOUT_S, json={"trip_id": trip_id})