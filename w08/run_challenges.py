"""
Week 8 -- does feature selection pay off (A3), and which selector (B1, B2)? Runs tuned XGBoost through main.run, the
exact code path of `python main.py`, once without selection and once per selector. Every run lands on the scoreboard
(results/runs.csv); the numbers for the README also go to w08/challenge_results.json (a generated, git-ignored file).

Run from the repo root:
    python w08/run_challenges.py                  # every scenario
    python w08/run_challenges.py off vote         # only A3

Same data, cleaning and folds for every run, so the rows are paired by fold (src/evaluate.py:corrected_paired_test).
Everything is sequential, so `seconds` compares the cost of the selectors.

`vote_2` is a check: with min_votes fixed at 2 instead of tuned, nested CV measures "keep what two voters agree on",
the setting the final vote model ended up with. It goes to its own scoreboard so results/runs.csv keeps one row per
selector, and the script also records which features the voters keep on each outer fold's training rows.
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
from src.evaluate import build_cv  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402
from src.selection import VoteSelector  # noqa: E402
from wk06.run_challenges import summarize  # noqa: E402


def scenario(base: dict, name: str) -> dict:
    config = copy.deepcopy(base)
    config["model"]["type"] = "xgboost"
    config["tuning"]["enabled"] = True
    selection = config["feature_selection"]
    selection["enabled"] = name != "off"
    if name == "vote_2":
        selection.update(method="vote", min_votes=2)
        selection["search_space"]["vote"] = {}
        config["output"]["scoreboard"] = "results/runs_week8_fixed_votes.csv"
    elif name != "off":
        selection["method"] = name
    return config


def votes_per_fold(base: dict) -> list:
    """For each outer fold: the features with 1, 2 and 3 votes, fitted on that fold's training rows only."""
    X_dev, y_dev, _ = load_development_set(base)
    folds = []
    for train_idx, _ in build_cv(base["cv"]).split(X_dev, y_dev):
        X_train, y_train = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        prep = build_preprocessor(base["preprocessing"]).fit(X_train, y_train)
        selector = VoteSelector(random_state=base["feature_selection"]["random_state"]).fit(prep.transform(X_train), y_train)
        names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
        folds.append({n: int(v) for n, v in zip(names, selector.votes_)})
    return folds


def main(names):
    os.chdir(ROOT)            # relative paths, exactly as `python main.py` uses them (the scoreboard matches runs by data path)
    base = load_config("config.yaml")
    _, y_dev, extras_dev = load_development_set(base)

    out_path = ROOT / "w08" / "challenge_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        t0 = time.time()
        if name == "votes_per_fold":
            results[name] = votes_per_fold(base)
        else:
            config = scenario(base, name)
            _, s = run(config)
            results[name] = summarize(config, s, y_dev, extras_dev)
            results[name].update(run=s["run"], kept_features=s["kept_features"], seconds=round(time.time() - t0, 1))
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n### {name}: done in {time.time() - t0:.1f}s\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["off", "vote", "kbest", "from_model", "vote_2", "votes_per_fold"])
