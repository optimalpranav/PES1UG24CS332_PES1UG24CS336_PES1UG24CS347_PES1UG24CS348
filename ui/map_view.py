"""Folium map of one /predict response: road network, predicted route, GPS points."""
import folium

MAX_GPS_POINTS = 2000


def build_map(res):
    gps = res["gps"]
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)],
                   zoom_start=15, tiles="OpenStreetMap")

    net = folium.FeatureGroup(name="Road network")
    if res.get("network"):
        # one multi-polyline: much lighter than one PolyLine per arc
        folium.PolyLine(res["network"], color="#888888", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    pred = folium.FeatureGroup(name="Predicted route")
    if res.get("predicted_route"):
        folium.PolyLine(res["predicted_route"], color="red", weight=4, opacity=0.9).add_to(pred)
    pred.add_to(m)

    pts = folium.FeatureGroup(name="GPS points")
    step = max(1, len(gps) // MAX_GPS_POINTS)
    for lat, lon in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, fill_opacity=0.7).add_to(pts)
    pts.add_to(m)

    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(m)
    return m