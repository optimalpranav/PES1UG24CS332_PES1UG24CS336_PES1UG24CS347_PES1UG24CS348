"""
Member 2: Bayesian hyperparameter optimisation (Optuna TPE) for the chosen model.
Objective: mean observation-level Top-1 accuracy over the 5 trip-grouped CV folds
(Top-1 is what the route decoder consumes). Student-test is used only after tuning.

Run from repo root:  python -m tuning.optuna_tune --trials 30
Outputs: tuning/trials.csv, tuning/best_params.json, models/saved/final_model.joblib,
         results/tuned_vs_untuned.csv, results/plots/optuna_history.png
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features.candidates import FEATURE_COLS
from models.train_compare import SEED, evaluate


def make_model(p):
    return HistGradientBoostingClassifier(random_state=SEED, **p)


def cv_score(params, train):
    scores = []
    for k in sorted(train["fold"].unique()):
        tr, va = train[train.fold != k], train[train.fold == k]
        m = make_model(params).fit(tr[FEATURE_COLS].to_numpy(), tr["label"])
        scores.append(evaluate(m, va)[0])
    return pd.DataFrame(scores).mean()


def main(n_trials):
    train = pd.read_parquet("features/out/train.parquet").merge(
        pd.read_csv("results/folds.csv", dtype={"trip_id": str}), on="trip_id")
    test = pd.read_parquet("features/out/student_test.parquet")

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
        s = cv_score(params, train)
        trial.set_user_attr("cv_macro_f1", float(s.macro_f1))
        return float(s.top1_acc)

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    tune_s = time.perf_counter() - t0
    study.trials_dataframe().to_csv("tuning/trials.csv", index=False)
    json.dump({"best_params": study.best_params, "best_cv_top1": study.best_value,
               "n_trials": n_trials, "tuning_time_s": tune_s},
              open("tuning/best_params.json", "w"), indent=2)

    rows = []
    for name, params in [("untuned", {"max_iter": 300, "learning_rate": 0.1}),
                         ("tuned", study.best_params)]:
        cv = cv_score(params, train)
        m = make_model(params).fit(train[FEATURE_COLS].to_numpy(), train["label"])
        te, _ = evaluate(m, test)
        rows.append({"model": name, "cv_macro_f1": cv.macro_f1, "cv_top1_acc": cv.top1_acc,
                     "test_top1_acc": te["top1_acc"], "test_macro_f1": te["macro_f1"],
                     "test_log_loss": te["log_loss"], "infer_ms_per_point": te["infer_ms_per_point"]})
        if name == "tuned":
            joblib.dump({"model": m, "features": FEATURE_COLS, "params": params},
                        "models/saved/final_model.joblib")
    res = pd.DataFrame(rows)
    res.to_csv("results/tuned_vs_untuned.csv", index=False)
    print(res)

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig("results/plots/optuna_history.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    main(ap.parse_args().trials)