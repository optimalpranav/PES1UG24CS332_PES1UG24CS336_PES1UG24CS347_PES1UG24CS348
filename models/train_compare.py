"""
Member 2: train and compare >= 5 models on the candidate table.

Task: binary classification of each (GPS point, candidate arc) row:
is this the reference arc? Top-1 per point = candidate with highest probability.
CV: 5 folds grouped by trip (results/folds.csv from Member 1).
Final fit: all 80 train trips -> evaluate on the 10 student_test trips.

Run from repo root:  python -m models.train_compare
Outputs: results/model_comparison.csv, results/plots/*.png, models/saved/*.joblib
"""
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features.candidates import FEATURE_COLS
from data_processing.load import load_trip
from utils.metrics import route_continuity, top1_accuracy, top1_predictions

SEED = 42
SAVE_DIR = Path("models/saved")
PLOT_DIR = Path("results/plots")


def get_models():
    return {
        "LogReg": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
        "GaussianNB": ("Probabilistic", make_pipeline(StandardScaler(), GaussianNB())),
        "KNN": ("Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
        "RandomForest": ("Ensemble (bagging)", RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "HistGB": ("Boosting", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)),
        "MLP": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                                                 max_iter=200, random_state=SEED))),
    }


def evaluate(model, df):
    X = df[FEATURE_COLS].to_numpy()
    t0 = time.perf_counter()
    prob = model.predict_proba(X)[:, 1]
    infer_ms = (time.perf_counter() - t0) / df.groupby(["trip_id", "obs_idx"]).ngroups * 1000
    pred = (prob >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(df["label"], pred, average="macro"),
        "log_loss": log_loss(df["label"], np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "top1_acc": top1_accuracy(df, top1_predictions(df, prob)),
        "infer_ms_per_point": infer_ms,
    }, prob


def test_continuity(test, prob):
    """Route continuity of classifier-only Top-1 predictions, averaged over student_test trips."""
    pred = top1_predictions(test, prob)
    vals = []
    for tid, g in pred.groupby("trip_id"):
        arcs = load_trip(Path("data/student_test") / tid, load_route=False)["arcs"]
        vals.append(route_continuity(g.sort_values("obs_idx")["pred_arc"].to_numpy(), arcs))
    return float(np.nanmean(vals))


def main():
    train = pd.read_parquet("features/out/train.parquet")
    test = pd.read_parquet("features/out/student_test.parquet")
    folds = pd.read_csv("results/folds.csv", dtype={"trip_id": str})
    train = train.merge(folds, on="trip_id")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (family, model) in get_models().items():
        cv = []
        for k in sorted(train["fold"].unique()):
            tr, va = train[train.fold != k], train[train.fold == k]
            model.fit(tr[FEATURE_COLS].to_numpy(), tr["label"])
            cv.append(evaluate(model, va)[0])
        cv = pd.DataFrame(cv)

        t0 = time.perf_counter()
        model.fit(train[FEATURE_COLS].to_numpy(), train["label"])
        train_s = time.perf_counter() - t0
        tr_res, _ = evaluate(model, train)
        te_res, te_prob = evaluate(model, test)
        joblib.dump(model, SAVE_DIR / f"{name}.joblib")

        rows.append({"model": name, "family": family, "config": str(model.steps[-1][1] if hasattr(model, "steps") else model),
                     "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
                     "cv_top1_acc": cv.top1_acc.mean(), "train_macro_f1": tr_res["macro_f1"],
                     "test_macro_f1": te_res["macro_f1"], "test_top1_acc": te_res["top1_acc"],
                     "test_log_loss": te_res["log_loss"],
                     "test_route_continuity": test_continuity(test, te_prob), "train_time_s": train_s,
                     "infer_ms_per_point": te_res["infer_ms_per_point"]})
        print(rows[-1])
        pd.DataFrame(rows).to_csv("results/model_comparison.csv", index=False)   # save after every model

        ConfusionMatrixDisplay.from_predictions(test["label"], (te_prob >= 0.5).astype(int))
        plt.title(f"{name}: student_test confusion matrix")
        plt.savefig(PLOT_DIR / f"cm_{name}.png", dpi=120, bbox_inches="tight")
        plt.close()

        if name == "MLP":   # only model with an epoch-wise history
            mlp = model.steps[-1][1]
            plt.plot(mlp.loss_curve_, label="training loss")
            plt.plot(1 - np.array(mlp.validation_scores_), label="validation error (1 - accuracy)")
            plt.xlabel("epoch"); plt.legend(); plt.title("MLP training history")
            plt.savefig(PLOT_DIR / "mlp_history.png", dpi=120, bbox_inches="tight")
            plt.close()

    res = pd.DataFrame(rows)
    res.to_csv("results/model_comparison.csv", index=False)

    for col, title in [("cv_macro_f1", "CV macro F1 (5 folds by trip)"),
                       ("test_top1_acc", "Student-test Top-1 accuracy"),
                       ("test_macro_f1", "Student-test macro F1"),
                       ("test_log_loss", "Student-test log loss (lower is better)")]:
        plt.figure(figsize=(7, 3.5))
        plt.bar(res["model"], res[col], yerr=res["cv_macro_f1_std"] if col == "cv_macro_f1" else None)
        plt.title(title); plt.xticks(rotation=30)
        plt.savefig(PLOT_DIR / f"{col}.png", dpi=120, bbox_inches="tight")
        plt.close()
    print(res[["model", "cv_macro_f1", "test_top1_acc", "test_macro_f1", "test_log_loss", "test_route_continuity"]])


if __name__ == "__main__":
    main()