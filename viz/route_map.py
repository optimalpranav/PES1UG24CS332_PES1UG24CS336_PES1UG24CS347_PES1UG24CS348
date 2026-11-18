"""
Member 4: Folium map of one trip with toggleable layers.
  - Road network (grey), clipped to the area around the GPS track
  - Reference route (green)
  - GPS points (blue), start/end markers
Run from repo root:  python -m viz.route_map 00000005 00000021
Output: viz/maps/trip_<id>.html
"""
import sys
from pathlib import Path

import folium

from data_processing.load import load_trip

TRAIN_DIR = Path("data/train")
OUT_DIR = Path("viz/maps")
MARGIN_DEG = 0.003      # ~300 m padding around the track bbox
MAX_GPS_POINTS = 2000   # thin long tracks so the HTML stays light


def arc_segments(nodes, arcs, arc_ids):
    """List of [[lat, lon], [lat, lon]] segments (Folium wants lat first)."""
    a = arcs.loc[arc_ids]
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    return [
        [[lat[f], lon[f]], [lat[t], lon[t]]]
        for f, t in zip(a["from_node"], a["to_node"])
    ]


def network_near_track(track, nodes, arcs):
    """Arc ids with at least one endpoint inside the track's padded bounding box."""
    lo_min, lo_max = track["lon"].min() - MARGIN_DEG, track["lon"].max() + MARGIN_DEG
    la_min, la_max = track["lat"].min() - MARGIN_DEG, track["lat"].max() + MARGIN_DEG
    inside = (
        nodes["lon"].between(lo_min, lo_max) & nodes["lat"].between(la_min, la_max)
    ).to_numpy()
    keep = inside[arcs["from_node"]] | inside[arcs["to_node"]]
    return arcs.index[keep]


def build_map(trip_id):
    trip = load_trip(TRAIN_DIR / trip_id)
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]

    center = [track["lat"].mean(), track["lon"].mean()]
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    # Layer 1: road network
    net = folium.FeatureGroup(name="Road network", show=True)
    for seg in arc_segments(nodes, arcs, network_near_track(track, nodes, arcs)):
        folium.PolyLine(seg, color="#888888", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    # Layer 2: reference route
    if route is not None:
        ref = folium.FeatureGroup(name="Reference route", show=True)
        for seg in arc_segments(nodes, arcs, route["arc_id"].to_numpy()):
            folium.PolyLine(seg, color="green", weight=5, opacity=0.8).add_to(ref)
        ref.add_to(m)

    # Layer 3: GPS points
    gps = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(track) // MAX_GPS_POINTS)
    for _, r in track.iloc[::step].iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=2, color="blue", fill=True,
            fill_opacity=0.7, weight=0, tooltip=f"t={r['t']:.0f}s",
        ).add_to(gps)
    folium.Marker([track["lat"].iloc[0], track["lon"].iloc[0]],
                  tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(gps)
    folium.Marker([track["lat"].iloc[-1], track["lon"].iloc[-1]],
                  tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(gps)
    gps.add_to(m)

    # Legend
    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
         background: white; padding: 10px 14px; border: 1px solid #999;
         border-radius: 6px; font-size: 13px;">
      <b>Trip {tid}</b><br>
      <span style="color:#888">&#9644;</span> Road network<br>
      <span style="color:green">&#9644;</span> Reference route<br>
      <span style="color:blue">&#9679;</span> GPS observations
    </div>""".format(tid=trip_id)
    m.get_root().html.add_child(folium.Element(legend))

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track["lat"].min(), track["lon"].min()],
                  [track["lat"].max(), track["lon"].max()]])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"trip_{trip_id}.html"
    m.save(str(out))
    print(f"Saved {out}")


if __name__ == "__main__":
    ids = sys.argv[1:] or ["00000005"]
    for tid in ids:
        build_map(tid)