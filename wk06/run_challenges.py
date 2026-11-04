"""
Week 6 -- ensembles and the challenges. Runs the pipeline (main.run, the exact code path of `python main.py`) once
per scenario and collects the numbers for the README into wk06/challenge_results.json (a generated, git-ignored file).

Week 6 compared models on ACCURACY (week 7 switches the pipeline to ROC-AUC), so every scenario here sets
cv.selection_metric: accuracy. The runs go to their own scoreboard, results/runs_week6_accuracy.csv.

Run from the repo root:
    python wk06/run_challenges.py                 # every scenario
    python wk06/run_challenges.py rf ada_fe       # only some

Every scenario keeps the rules: development set only, whole pipeline, same outer folds (cv) and inner folds
(tuning), nested CV for anything tuned. Only config values change -- never the code.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import load_config, load_development_set, run  # noqa: E402
from src.evaluate import build_cv, build_scorers, cross_validate_pipeline  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402

DEFAULT_PARAMS = {
    "logistic_regression": {"max_iter": 2000},
    "decision_tree": {"random_state": 42},
    "random_forest": {"n_estimators": 200, "random_state": 42, "n_jobs": 1},
    "oblique_random_forest": {"n_estimators": 200, "n_components": 10, "random_state": 42, "n_jobs": 1},
    "adaboost": {"max_depth": 1, "n_estimators": 200, "learning_rate": 1.0, "random_state": 42},
}
FEATURES = ["priors_per_adult_year", "has_juvenile_record", "young_x_priors", "priors_bin"]


def scenario(base: dict, model_type: str, features: list = None) -> dict:
    config = copy.deepcopy(base)
    config["model"] = {"type": model_type, "params": DEFAULT_PARAMS[model_type]}
    config["cv"]["selection_metric"] = "accuracy"
    config["preprocessing"]["engineered_features"] = features or []
    config["feature_selection"]["enabled"] = False
    config["output"]["scoreboard"] = "results/runs_week6_accuracy.csv"
    return config


SCENARIOS = {
    # the README "Ensembles" table (logistic regression tuned too: challenge 1 needs its nested score)
    "lr": lambda c: scenario(c, "logistic_regression"),
    "tree": lambda c: scenario(c, "decision_tree"),
    "rf": lambda c: scenario(c, "random_forest"),
    "orf": lambda c: scenario(c, "oblique_random_forest"),
    "ada": lambda c: scenario(c, "adaboost"),
    # challenge 1: the same three model families with the engineered features
    "lr_fe": lambda c: scenario(c, "logistic_regression", FEATURES),
    "rf_fe": lambda c: scenario(c, "random_forest", FEATURES),
    "ada_fe": lambda c: scenario(c, "adaboost", FEATURES),
}


def summarize(config, s, y_dev, extras_dev) -> dict:
    nested, study = s["nested_scores"], s["study"]
    default = s["default_scores"]
    known = {"fold", "train", "validation", "gap", "inner_best", *config["cv"]["metrics"]}
    param_cols = [c for c in nested.columns if c not in known]
    y_pred = (s["oof_proba"] >= 0.5).astype(int)
    negatives = np.asarray(y_dev) == 0
    race = np.asarray(extras_dev["race"], dtype=object)
    return {
        "model": config["model"]["type"],
        "metric": config["cv"]["selection_metric"],
        "features": config["preprocessing"]["engineered_features"],
        "default_cv": [float(default["validation"].mean()), float(default["validation"].std(ddof=1))],
        "default_gap": float(default["gap"].mean()),
        "nested_cv": [float(nested["validation"].mean()), float(nested["validation"].std(ddof=1))],
        "nested_gap": float(nested["gap"].mean()),
        "nested_folds": nested["validation"].tolist(),
        "default_folds": default["validation"].tolist(),
        "nested_metrics": {m: float(nested[m].mean()) for m in config["cv"]["metrics"]},
        "inner_best_mean": float(nested["inner_best"].mean()),
        "optimism": float(nested["inner_best"].mean() - nested["validation"].mean()),
        "best_trial_score": float(study.best_value),
        "best_params": study.best_params,
        "per_fold_params": nested[param_cols].astype(object).where(nested[param_cols].notna(), None)
                                             .to_dict(orient="records"),
        "oof_recall_pos": float(recall_score(y_dev, y_pred)),
        "oof_fpr": {g: float(y_pred[negatives & (race == g)].mean()) for g in ["African-American", "Caucasian"]},
        "results_file": Path(s["results_path"]).name,
    }


def boost_logistic_regression(base: dict) -> dict:
    """Challenge 2: AdaBoost over logistic regressions vs one logistic regression, same folds, default settings."""
    config = scenario(base, "logistic_regression")
    X_dev, y_dev, _ = load_development_set(config)
    cv, metrics = build_cv(config["cv"]), config["cv"]["metrics"]
    scorers = build_scorers(metrics)
    out = {}
    for name, model in [
        ("logistic_regression", LogisticRegression(max_iter=2000)),
        ("adaboost_lr_50", AdaBoostClassifier(estimator=LogisticRegression(max_iter=2000), n_estimators=50, random_state=42)),
        ("adaboost_lr_200_lr0.1", AdaBoostClassifier(estimator=LogisticRegression(max_iter=2000), n_estimators=200,
                                                     learning_rate=0.1, random_state=42)),
    ]:
        pipe = Pipeline([("prep", build_preprocessor(config["preprocessing"])), ("model", model)])
        fs, proba = cross_validate_pipeline(pipe, X_dev, y_dev, cv, scorers, "accuracy", config["cv"]["n_jobs"])
        fitted = pipe.fit(X_dev, y_dev).named_steps["model"]
        out[name] = {m: [float(fs[m].mean()), float(fs[m].std(ddof=1))] for m in ["accuracy", "roc_auc"]}
        out[name]["folds_accuracy"] = fs["accuracy"].tolist()
        out[name]["folds_roc_auc"] = fs["roc_auc"].tolist()
        out[name]["distinct_probabilities"] = int(len(np.unique(np.round(proba, 6))))
        if isinstance(fitted, AdaBoostClassifier):
            out[name]["rounds_kept"] = len(fitted.estimators_)
            out[name]["estimator_weights_first5"] = [float(w) for w in fitted.estimator_weights_[:5]]
    return out


def main(names):
    os.chdir(ROOT)            # relative paths, exactly as `python main.py` uses them (the scoreboard matches runs by data path)
    base = load_config("config.yaml")
    _, y_dev, extras_dev = load_development_set(base)

    out_path = ROOT / "wk06" / "challenge_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        t0 = time.time()
        if name == "boost_lr":
            results[name] = boost_logistic_regression(base)
        else:
            config = SCENARIOS[name](base)
            _, s = run(config)
            results[name] = summarize(config, s, y_dev, extras_dev)
        results[name]["seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n### {name}: done in {results[name]['seconds']}s\n", flush=True)



if __name__ == "__main__":
    main(sys.argv[1:] or [*SCENARIOS, "boost_lr"])
