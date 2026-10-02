"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (accuracy, classification report, fairness table) to a timestamped file in `results/` instead, so youcan open it again later, or compare two runs side by side after changing something in config.yaml.
"""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    prep, evaluation = config["preprocessing"], config["evaluation"]
    if evaluation["method"] == "cv":
        cv = config["cv"]
        method = (f"cross-validation  n_splits={cv['n_splits']}  shuffle={cv.get('shuffle', True)}  "
                  f"random_state={cv.get('random_state')}  scoring={cv.get('scoring', 'accuracy')}")
    else:
        method = (f"holdout  validation_size={evaluation['validation_size']}  "
                  f"random_state={evaluation['random_state']}")

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={prep['encoder']}  scaler={prep['scaler']}  "
        f"numeric_imputer={prep.get('numeric_imputer', 'median')}  log_features={prep.get('log_features') or []}\n"
        f"Evaluation: {method}  (development set only)\n"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  -- not scored\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
