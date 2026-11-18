# PathSnap UI Demo

This folder provides a **stand‑alone Gradio dashboard** that demonstrates the PathSnap UI layout using static demo data. It does **not** connect to any backend; all visualisations and metrics are hard‑coded for demonstration purposes.

## Files
- `app.py` – entry point for the Gradio app.
- `map_visualization.py` – helper functions that build a Folium map from a JSON‑like response.
- `requirements.txt` – Python dependencies.
- `README.md` – this file.

## Setup
```bash
# (optional) create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Run the UI
```bash
python app.py
```
A browser window will open at `http://127.0.0.1:7860` displaying the demo dashboard.

## What the demo shows
1. **Dataset Overview** – description and a dropdown with ten dummy trip IDs.
2. **Route Reconstruction** – clicking *Reconstruct Route (demo)* renders a Folium map with:
   - Blue GPS observation markers
   - Grey road‑network line segments
   - Green predicted route line
   - Orange reference route line (optional)
3. **Model Performance** – a table of example metrics (not real model results).

## Next steps
When the real FastAPI backend becomes available, replace the demo callbacks in `app.py` with calls to an `api_client.py` module that talks to the backend, and remove the static `DEMO_RESPONSE` data.

---
*This UI code is written for the PathSnap hackathon project and may be freely modified under the project’s licensing terms.*
