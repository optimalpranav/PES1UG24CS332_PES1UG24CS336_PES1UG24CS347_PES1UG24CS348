# PathSnap UI 

Streamlit app for the evaluator. It talks to the PathSnap FastAPI backend
running on Laptop 1:

1. Lists the evaluation trips from `GET /trips`.
2. Sends the selected trip ID to `POST /predict`.
3. Shows the predicted route on a Folium map, together with the metrics
   (Top-1 accuracy, route continuity, classifier-only vs HMM/Viterbi)
   computed **server-side** by the API.

The UI never receives the reference route. All prediction and scoring
happen on Laptop 1.

## Files
- `app.py` – Streamlit entry point (trip selection, metrics, map)
- `api_client.py` – HTTP calls to the API (`/health`, `/trips`, `/predict`)
- `map_view.py` – builds the Folium map from a `/predict` response
  (road network, predicted route, GPS points, start/end markers)
- `requirements.txt` – UI dependencies

## Setup
```bash
pip install -r ui/requirements.txt
```

## Run
Start the API on Laptop 1 first (see the main README), then on Laptop 2,
from the repo root:
```bash
streamlit run ui/app.py
```
The app opens in the browser at `http://localhost:8501`.

## Connecting to Laptop 1
Enter the API address in the sidebar, e.g. `http://192.168.1.20:8000`
(use Laptop 1's IPv4 address from `ipconfig`), or set it in advance:
```bash
set PATHSNAP_API_URL=http://192.168.1.20:8000
```
The sidebar shows **Connected** with the loaded model and decoder
(`viterbi` when the HMM is active).

## Map legend
- Blue dots – GPS observations
- Red line – predicted route
- Grey lines – road network around the trip
- Green / red markers – start / end of the trip