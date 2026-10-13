"""
Week 5 challenges -- runs the pipeline (main.run, the exact code path of `python main.py`) once per scenario
and collects the numbers for the README table into wk05/challenge_results.json.

Run from the repo root:
    python wk05/run_challenges.py              # every scenario
    python wk05/run_challenges.py tree lr      # only some

Every scenario keeps the rules: development set only, whole pipeline, same outer folds (cv) and same inner
folds (tuning.n_splits / tuning.random_state). Only config values change -- never the code.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import load_config, load_development_set, run  # noqa: E402

DEFAULT_PARAMS = {
    "decision_tree": {"random_state": 42},
    "logistic_regression": {"max_iter": 2000},
    "random_forest": {"random_state": 42, "n_jobs": 1},
}


def scenario(base: dict, model_type: str, n_trials: int, scoring: str = None, extra_space: dict = None) -> dict:
    config = copy.deepcopy(base)
    config["model"] = {"type": model_type, "params": DEFAULT_PARAMS[model_type]}
    config["tuning"]["n_trials"] = n_trials
    if scoring:
        config["cv"]["scoring"] = scoring
    if extra_space:
        config["tuning"]["search_spaces"][model_type].update(extra_space)
    return config


SCENARIOS = {
    # challenge 1 + 2: the README table (plus the tree from the notebook)
    "tree": lambda c: scenario(c, "decision_tree", 30),
    "lr": lambda c: scenario(c, "logistic_regression", 30),
    "rf": lambda c: scenario(c, "random_forest", 15),
    # challenge 3: tune the numeric imputer together with the tree
    "tree_impute": lambda c: scenario(c, "decision_tree", 30,
                                      extra_space={"prep__numeric__impute__strategy":
                                                   {"type": "categorical", "choices": ["median", "mean"]}}),
    # challenge 4: change what "best" means
    "tree_bal": lambda c: scenario(c, "decision_tree", 30, scoring="balanced_accuracy"),
    "lr_bal": lambda c: scenario(c, "logistic_regression", 30, scoring="balanced_accuracy"),
    "rf_bal": lambda c: scenario(c, "random_forest", 15, scoring="balanced_accuracy"),
}


def false_positive_rates(y_true, y_pred, groups) -> dict:
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups, dtype=object)
    out = {}
    for g in ["African-American", "Caucasian", "Hispanic", "Other"]:
        negatives = (groups == g) & (y_true == 0)
        out[g] = float((y_pred[negatives] == 1).mean())
    return out


def mean_std(series) -> list:
    return [float(series.mean()), float(series.std(ddof=1))]


def main(names):
    base = load_config(ROOT / "config.yaml")
    base["data"]["path"] = str(ROOT / base["data"]["path"])
    base["output"]["results_dir"] = str(ROOT / "results")
    _, y_dev, extras_dev = load_development_set(base)

    out_path = ROOT / "wk05" / "challenge_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        config = SCENARIOS[name](base)
        t0 = time.time()
        _, s = run(config)
        nested, study, y_oof = s["nested_scores"], s["study"], s["y_oof"]
        param_cols = [c for c in nested.columns if c not in {"fold", "train", "validation", "gap", "inner_best"}]
        results[name] = {
            "model": config["model"]["type"],
            "scoring": config["cv"]["scoring"],
            "n_trials": config["tuning"]["n_trials"],
            "default_cv": mean_std(s["default_scores"]["validation"]),
            "default_gap": float(s["default_scores"]["gap"].mean()),
            "nested_cv": mean_std(nested["validation"]),
            "nested_gap": float(nested["gap"].mean()),
            "inner_best_mean": float(nested["inner_best"].mean()),
            "optimism": float(nested["inner_best"].mean() - nested["validation"].mean()),
            "best_trial_score": float(study.best_value),
            "best_params": study.best_params,
            "per_fold_params": nested[param_cols].astype(object).where(nested[param_cols].notna(), None)
                                                 .to_dict(orient="records"),
            "oof_accuracy": float(accuracy_score(y_dev, y_oof)),
            "oof_balanced_accuracy": float(balanced_accuracy_score(y_dev, y_oof)),
            "oof_recall_pos": float(recall_score(y_dev, y_oof)),
            "oof_fpr": false_positive_rates(y_dev, y_oof, extras_dev["race"]),
            "seconds": round(time.time() - t0, 1),
            "results_file": Path(s["results_path"]).name,
        }
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n### {name}: done in {results[name]['seconds']}s\n")


if __name__ == "__main__":
    main(sys.argv[1:] or list(SCENARIOS))
