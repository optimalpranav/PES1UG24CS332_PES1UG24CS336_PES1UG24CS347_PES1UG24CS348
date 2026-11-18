import joblib

from api.config import MODEL_CANDIDATES
from features.candidates import FEATURE_COLS


def load_model():
    """final_model.joblib is a dict {model, features, params};
    the train_compare.py files are bare estimators. Handle both."""
    for path in MODEL_CANDIDATES:
        if not path.exists():
            continue
        obj = joblib.load(path)
        if isinstance(obj, dict):
            model, feats, params = obj["model"], list(obj.get("features", FEATURE_COLS)), obj.get("params")
        else:
            model, feats, params = obj, FEATURE_COLS, None
        if feats != FEATURE_COLS:
            raise RuntimeError(f"{path.name} was trained on {feats}, pipeline builds {FEATURE_COLS}")
        return {"model": model, "features": feats, "params": params,
                "name": path.stem, "path": str(path)}
    raise FileNotFoundError(f"No model found in {[str(p) for p in MODEL_CANDIDATES]}")