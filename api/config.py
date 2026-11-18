import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Trips the evaluator can pick. Defaults to student_test for development;
# on evaluation day point it at the reserved trips:  set PATHSNAP_EVAL_DIR=D:\...\eval_trips
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "student_test"))

# First one that exists is loaded: tuned model once Optuna finishes, else untuned HistGB
MODEL_CANDIDATES = [
    ROOT / "models" / "saved" / "final_model.joblib",
    ROOT / "models" / "saved" / "HistGB.joblib",
]

NETWORK_MARGIN_DEG = 0.003   # ~300 m road-network padding around the track (same as viz/route_map.py)