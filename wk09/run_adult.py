"""
Week 9 -- the Adult runs for the README. Runs main.run (the exact code path of `python main.py config_adult.yaml`)
once per scenario on the same folds; every run lands on results/adult/runs.csv, and the numbers for the README go to
wk09/adult_results.json (a generated, git-ignored file).

Run from the repo root:
    python wk09/run_adult.py                 # every scenario
    python wk09/run_adult.py xgb_vote        # only some

Only `xgb_vote` -- exactly config_adult.yaml -- writes the predictions file; the other scenarios skip that step.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import load_config, load_development_set, run  # noqa: E402

PARAMS = {
    "decision_tree": {"random_state": 42},
    "logistic_regression": {"max_iter": 2000},
    "hist_gradient_boosting": {"random_state": 42},
}
SCENARIOS = {
    # name: (model type, tuned, feature selection method or None)
    "tree_default": ("decision_tree", False, None),
    "lr": ("logistic_regression", True, None),
    "hgb": ("hist_gradient_boosting", True, None),
    "xgb_off": ("xgboost", True, None),
    "xgb_vote": ("xgboost", True, "vote"),          # = config_adult.yaml
    "xgb_kbest": ("xgboost", True, "kbest"),
    "xgb_from_model": ("xgboost", True, "from_model"),
}


def scenario(base: dict, name: str) -> dict:
    model_type, tuned, method = SCENARIOS[name]
    config = copy.deepcopy(base)
    if model_type != "xgboost":
        config["model"] = {"type": model_type, "params": PARAMS[model_type]}
    config["tuning"]["enabled"] = tuned
    config["feature_selection"]["enabled"] = method is not None
    if method:
        config["feature_selection"]["method"] = method
    if name != "xgb_vote":
        config.pop("predict", None)
    return config


def summarize(config, s, y_dev, extras_dev) -> dict:
    scores = s.get("nested_scores", s["default_scores"])
    metrics = config["cv"]["metrics"]
    known = {"fold", "train", "validation", "gap", "inner_best", *metrics}
    y_pred = (s["oof_proba"] >= 0.5).astype(int)
    negatives = np.asarray(y_dev) == 0
    group = np.asarray(extras_dev[config["data"]["sensitive_attr"]], dtype=object)
    out = {
        "model": config["model"]["type"], "tuned": "nested_scores" in s,
        "feature_selection": config["feature_selection"]["method"] if config["feature_selection"]["enabled"] else "off",
        "default": {m: [float(s["default_scores"][m].mean()), float(s["default_scores"][m].std(ddof=1))] for m in metrics},
        "default_gap": float(s["default_scores"]["gap"].mean()),
        "final": {m: [float(scores[m].mean()), float(scores[m].std(ddof=1))] for m in metrics},
        "final_gap": float(scores["gap"].mean()),
        "folds": scores["validation"].tolist(),
        "kept_features": s["kept_features"],
        "fpr": {g: float(y_pred[negatives & (group == g)].mean()) for g in sorted(set(group))},
        "run": s["run"],
    }
    if "study" in s:
        out["best_params"] = s["study"].best_params
        out["best_trial"] = float(s["study"].best_value)
        out["optimism"] = float(scores["inner_best"].mean() - scores["validation"].mean())
        out["per_fold_params"] = scores[[c for c in scores.columns if c not in known]].astype(object).to_dict(orient="records")
    return out


def main(names):
    os.chdir(ROOT)
    base = load_config("config_adult.yaml")
    _, y_dev, extras_dev = load_development_set(base)
    out_path = ROOT / "wk09" / "adult_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        t0 = time.time()
        config = scenario(base, name)
        _, s = run(config)
        results[name] = summarize(config, s, y_dev, extras_dev)
        results[name]["seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n### {name}: done in {results[name]['seconds']}s\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SCENARIOS))
