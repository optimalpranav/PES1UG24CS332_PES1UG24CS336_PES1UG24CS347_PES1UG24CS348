"""
PathSnap evaluator UI (Laptop 2).
Run from repo root:  streamlit run ui/app.py
"""
import time

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from api_client import DEFAULT_API_URL, ApiError, health, list_trips, predict
from map_view import build_map

PCT = {"top1_acc": "Top-1 accuracy", "route_continuity": "Route continuity"}
COUNT = {"n_points": "GPS points", "points_with_candidates": "Points with candidates"}

st.set_page_config(page_title="PathSnap Evaluator", layout="wide")
st.title("PathSnap: Route Reconstruction")

# ---- API connection ----
with st.sidebar:
    st.header("API (Laptop 1)")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    try:
        h = health(api_url)
        st.success(f"Connected\n\nModel: {h['model']}\n\nDecoder: {h['decoder']}")
    except ApiError as e:
        st.error(str(e))
        st.stop()

# ---- trip selection ----
trips = list_trips(api_url)
st.subheader(f"Evaluation trips ({len(trips)})")
trip_id = st.radio("Select a trip", trips, horizontal=True)

if st.button("Reconstruct route", type="primary"):
    with st.spinner(f"Sending trip {trip_id} to the API..."):
        t0 = time.perf_counter()
        try:
            st.session_state["res"] = predict(api_url, trip_id)
            st.session_state["rtt_ms"] = (time.perf_counter() - t0) * 1000
        except ApiError as e:
            st.error(str(e))

# ---- results (kept in session_state so the map survives reruns) ----
res = st.session_state.get("res")
if res:
    m = res["metrics"]
    st.subheader(f"Trip {res['trip_id']}")
    st.caption(f"Model: {res['model']}  |  Decoder: {res['decoder']}  |  "
               f"Round trip: {st.session_state['rtt_ms']:.0f} ms")

    shown = [(k, f"{m[k] * 100:.2f}%") for k in PCT if k in m] + \
            [(k, f"{m[k]}") for k in COUNT if k in m]
    for col, (k, v) in zip(st.columns(len(shown)), shown):
        col.metric(PCT.get(k) or COUNT.get(k), v)

    if "classifier_only" in m:   # appears once the API uses Viterbi
        st.markdown("**Classifier-only vs HMM/Viterbi**")
        st.dataframe(pd.DataFrame({
            "Classifier only": m["classifier_only"],
            "HMM + Viterbi": {k: m[k] for k in m["classifier_only"]},
        }).T.rename(columns=PCT).style.format("{:.2%}"))

    left, right = st.columns([3, 1])
    with left:
        st_folium(build_map(res), height=600, use_container_width=True, returned_objects=[])
    with right:
        st.markdown("**Legend**\n\n:blue[●] GPS points  \n:red[━] Predicted route  \n:gray[━] Road network")
        st.markdown("**Server timing (ms)**")
        st.table(pd.Series(res["timing_ms"], name="ms"))