"""
Week 7 -- fill the scoreboard (A4) and the feature-selection challenge (B1). Runs the pipeline (main.run, the exact
code path of `python main.py`) once per model, tuned where it has a search space, on the selection metric of
config.yaml (ROC-AUC). Every run is added to results/runs.csv; the numbers for the README also go to
wk07/scoreboard_results.json (a generated, git-ignored file).

Run from the repo root:
    python wk07/run_scoreboard.py                       # every model, then B1
    python wk07/run_scoreboard.py gradient_boosting     # only some

Same data, cleaning and folds for every run, so the rows are paired by fold and can be compared with
src/evaluate.py:corrected_paired_test.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import load_config, load_development_set, run  # noqa: E402
from wk06.run_challenges import summarize  # noqa: E402

DEFAULT_PARAMS = {
    "dummy": {},
    "logistic_regression": {"max_iter": 2000},
    "naive_bayes": {},
    "decision_tree": {"random_state": 42},
    "random_forest": {"n_estimators": 200, "random_state": 42, "n_jobs": 1},
    "oblique_random_forest": {"n_estimators": 200, "n_components": 10, "random_state": 42, "n_jobs": 1},
    "adaboost": {"max_depth": 1, "n_estimators": 200, "learning_rate": 1.0, "random_state": 42},
    "gradient_boosting": {"random_state": 42},
    "hist_gradient_boosting": {"random_state": 42},
}
SELECTION_RUNS = {"select_logistic_regression": "logistic_regression", "select_gradient_boosting": "gradient_boosting"}


def scenario(base: dict, model_type: str, selection: bool = False) -> dict:
    config = copy.deepcopy(base)
    config["model"] = {"type": model_type, "params": DEFAULT_PARAMS[model_type]}
    config["tuning"]["enabled"] = model_type in config["tuning"]["search_spaces"]
    config["feature_selection"]["enabled"] = selection
    return config


def main(names):
    os.chdir(ROOT)            # relative paths, exactly as `python main.py` uses them (the scoreboard matches runs by data path)
    base = load_config("config.yaml")
    _, y_dev, extras_dev = load_development_set(base)

    out_path = ROOT / "wk07" / "scoreboard_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        t0 = time.time()
        config = (scenario(base, SELECTION_RUNS[name], selection=True) if name in SELECTION_RUNS
                  else scenario(base, name))
        _, s = run(config)
        if "study" in s:
            results[name] = summarize(config, s, y_dev, extras_dev)
        else:   # untuned (the dummy): default scores only
            d = s["default_scores"]
            results[name] = {"model": config["model"]["type"], "metric": config["cv"]["selection_metric"],
                             "default_cv": [float(d["validation"].mean()), float(d["validation"].std(ddof=1))],
                             "default_folds": d["validation"].tolist()}
        results[name]["run"] = s["run"]
        results[name]["seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n### {name}: done in {results[name]['seconds']}s\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or [*DEFAULT_PARAMS, *SELECTION_RUNS])
