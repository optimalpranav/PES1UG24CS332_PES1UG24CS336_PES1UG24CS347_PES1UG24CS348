import csv, json, os
from datetime import datetime

LOG_PATH = "results/experiments.csv"

def log_run(model_name, params, metrics, seed, notes=""):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "params": json.dumps(params),
        "seed": seed,
        **metrics,
        "notes": notes,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)