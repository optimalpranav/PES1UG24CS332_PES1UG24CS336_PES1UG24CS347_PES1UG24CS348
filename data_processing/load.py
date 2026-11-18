from pathlib import Path
import pandas as pd

def _read(path, names, dtypes):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtypes)

def load_trip(trip_dir, load_route=True):
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track = _read(trip_dir / f"{trip_id}.track", ["lon", "lat", "t"], dtypes={"lon": float, "lat": float, "t": float})

    nodes = _read(trip_dir / f"{trip_id}.nodes", ["lon", "lat", "t"], dtypes={"lon": float, "lat": float, "t": float})
    nodes.index.name = "node_id"

    arcs = _read(trip_dir / f"{trip_id}.arcs", ["from_node", "to_node"], dtypes={"from_node": int, "to_node": int})
    arcs.index.name = "arc_id"

    route = None
    route_path = trip_dir / f"{trip_id}.route"

    if load_route and route_path.exists():
        route = _read(route_path, ["arc_id"], {"arc_id": int})

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route
    }

def list_trips(partition_dir):
    return sorted(p for p in Path(partition_dir).iterdir() if p.is_dir())

if __name__ == "__main__":
    import sys 
    partition = sys.argv[1] if len(sys.argv) > 1 else "Student folder/train"
    trips = list_trips(partition)
    print(f"Found {len(trips)} trips in {partition}")
    t = load_trip(trips[0])
    print(t["trip_id"])
    for k in ["track", "nodes", "arcs", "route"]:
        df = t[k]
        print(f"{k:6s}",None if df is None else df.shape)
    print(t["track"].head(3))