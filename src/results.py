"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (per-fold tables, tuning, confusion matrix, fairness table) to a timestamped file in `results/`, so you can open it again later, or compare two runs side by side after changing something in config.yaml.

Week 7: `save_scores()` also adds one row per run to the scoreboard (`results/runs.csv`), so every run stays comparable with every other run on the same data, cleaning and folds.
"""
import os
from datetime import datetime

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to `results_dir/run_<run>.txt` (the folder is created if it
    doesn't exist yet) and returns the path that was written. `run` is the run id that main.py
    also puts on the scoreboard, so a scoreboard row and its report can be matched.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    prep, evaluation = config["preprocessing"], config["evaluation"]
    if evaluation["method"] == "cv":
        cv = config["cv"]
        method = (f"cross-validation  n_splits={cv['n_splits']}  shuffle={cv.get('shuffle', True)}  "
                  f"random_state={cv.get('random_state')}\n"
                  f"Metrics: {cv['metrics']}  selection_metric={cv['selection_metric']}")
    else:
        method = (f"holdout  validation_size={evaluation['validation_size']}  "
                  f"random_state={evaluation['random_state']}")

    tuning = config.get("tuning", {})
    if tuning.get("enabled", False):
        tuning_line = (f"Tuning: Optuna TPE  n_trials={tuning['n_trials']}  inner n_splits={tuning['n_splits']}  "
                       f"random_state={tuning['random_state']}  "
                       f"search_space={(tuning.get('search_spaces') or {}).get(config['model']['type'])}\n")
    else:
        tuning_line = "Tuning: off\n"
    selection = config.get("feature_selection") or {}
    if selection.get("enabled", False):
        settings = {key: value for key, value in selection.items() if key not in {"enabled", "search_space"}}
        selection_line = f"Feature selection: on  {settings}\n"
    else:
        selection_line = "Feature selection: off\n"

    header = (
        f"Run: {run}   (written {datetime.now():%Y-%m-%d %H:%M:%S})\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={prep['encoder']}  scaler={prep['scaler']}  "
        f"numeric_imputer={prep.get('numeric_imputer', 'median')}  log_features={prep.get('log_features') or []}  "
        f"engineered_features={prep.get('engineered_features') or []}\n"
        + selection_line +
        f"Evaluation: {method}  (development set only)\n"
        + tuning_line +
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  -- not scored\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path
