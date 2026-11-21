"""
Week 10 -- missing values: the challenges. Numbers for the README go to wk10/imputer_results.json (git-ignored).

    python wk10/run_imputers.py              # everything
    python wk10/run_imputers.py a1 b2        # only some parts

a1  every numeric imputer through main.run (the exact `python main.py` code path), tuning off, on the same folds:
    the configured model (XGBoost + vote selection) and a logistic regression. Runs land on results/runs.csv.
b1  multiple imputation: m pipelines with IterativeImputer(sample_posterior=True), probabilities averaged.
b2  the imputer also sees the categorical columns (one-hot, as context): guessing hidden values, and the model score.
b3  random-forest proximities from scratch, on hidden priors_count values.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import load_config, load_development_set, run  # noqa: E402
from src.evaluate import build_cv  # noqa: E402
from src.model import build_model, build_pipeline  # noqa: E402
from src.preprocessing import _NUMERIC_IMPUTERS, _SCALERS, build_preprocessor, mnar_columns  # noqa: E402
from src.selection import build_selector  # noqa: E402

IMPUTERS = ["median", "mean", "knn", "iterative", "random_forest"]
MODEL_IMPUTERS = ["knn", "iterative", "random_forest"]


# ------------------------------------------------------------------------ a1
def a1(base):
    out = {}
    for model_name in ["config_model", "logistic_regression"]:
        for imputer in IMPUTERS:
            config = copy.deepcopy(base)
            config["tuning"]["enabled"] = False
            config["preprocessing"]["numeric_imputer"] = imputer
            config["output"].pop("model_path", None)
            if model_name == "logistic_regression":
                config["model"] = {"type": "logistic_regression", "params": {"max_iter": 2000}}
                config["feature_selection"]["enabled"] = False
            t0 = time.time()
            _, s = run(config)
            d = s["default_scores"]
            out[f"{model_name}/{imputer}"] = {
                "roc_auc": [float(d["roc_auc"].mean()), float(d["roc_auc"].std(ddof=1))],
                "accuracy": float(d["accuracy"].mean()), "folds": d["roc_auc"].tolist(),
                "kept_features": s["kept_features"], "seconds": round(time.time() - t0, 1)}
            print(f"### a1 {model_name}/{imputer}: done", flush=True)
    return out


# ------------------------------------------------------------------------ b1
def b1(base, X, y, m=5):
    """Fold by fold: one pipeline per imputer setting, or m MICE draws averaged -- same folds as everything else."""
    cv = build_cv(base["cv"])
    prep, model, selection = base["preprocessing"], base["model"], base["feature_selection"]
    settings = {"median": None, "iterative (one dataset)": "single", f"MICE, m={m} averaged": "multiple"}
    out = {name: {"folds": [], "seconds": 0.0} for name in settings}
    draws_spread = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr, X_va, y_va = X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx]
        for name, kind in settings.items():
            t0 = time.time()
            if kind is None:
                proba = build_pipeline(prep, model, selection).fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
            elif kind == "single":
                p = build_pipeline({**prep, "numeric_imputer": "iterative"}, model, selection)
                proba = p.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
            else:
                probas, filled = [], []
                for seed in range(m):
                    p = build_pipeline(prep, model, selection)
                    p.set_params(prep__numeric__impute=IterativeImputer(sample_posterior=True, random_state=seed))
                    p.fit(X_tr, y_tr)
                    probas.append(p.predict_proba(X_va)[:, 1])
                    numeric = p["prep"].named_transformers_["numeric"]
                    gaps = X_va["priors_count"].isna()
                    values = numeric["scale"].inverse_transform(numeric.transform(X_va.loc[gaps, prep["numeric_features"]]))
                    filled.append(values[:, prep["numeric_features"].index("priors_count")])
                proba = np.mean(probas, axis=0)
                draws_spread.append(float(np.mean(np.std(filled, axis=0))))
            out[name]["folds"].append(float(roc_auc_score(y_va, proba)))
            out[name]["seconds"] += time.time() - t0
    for name in out:
        out[name]["roc_auc"] = [float(np.mean(out[name]["folds"])), float(np.std(out[name]["folds"], ddof=1))]
        out[name]["seconds"] = round(out[name]["seconds"], 1)
    out["priors_draw_std"] = float(np.mean(draws_spread))      # how much the m datasets disagree about a missing priors_count
    print("### b1: done", flush=True)
    return out


# ------------------------------------------------------------------------ b2
class ContextImputer(BaseEstimator, TransformerMixin):
    """
    Scales the numeric columns, then imputes them with `imputer` while it also SEES the categorical columns
    (one-hot encoded, gaps filled with the most frequent value). Returns only the numeric columns, imputed and scaled:
    the categorical columns are context for the imputer, not new features.
    """

    def __init__(self, numeric=None, context=None, imputer=None, scaler=None):
        self.numeric, self.context, self.imputer, self.scaler = numeric, context, imputer, scaler

    def _context(self, X, fit):
        if fit:
            self.context_ = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
            return self.context_.fit_transform(X[self.context])
        return self.context_.transform(X[self.context])

    def fit(self, X, y=None):
        self.scaler_ = clone(self.scaler).fit(X[self.numeric])
        self.imputer_ = clone(self.imputer).fit(np.hstack([self.scaler_.transform(X[self.numeric]), self._context(X, True)]))
        return self

    def transform(self, X):
        both = np.hstack([self.scaler_.transform(X[self.numeric]), self._context(X, False)])
        return self.imputer_.transform(both)[:, :len(self.numeric)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.numeric, dtype=object)


def context_pipeline(config, imputer_name):
    prep = config["preprocessing"]
    numeric, categorical = prep["numeric_features"], prep["categorical_features"]
    imputer = _NUMERIC_IMPUTERS[imputer_name](prep.get("imputer_settings") or {}, prep.get("random_state"))
    standard = build_preprocessor(prep)               # reuse its categorical and indicator blocks unchanged
    blocks = dict((name, (step, cols)) for name, step, cols in standard.transformers)
    prep_step = ColumnTransformer([
        ("numeric", ContextImputer(numeric, categorical, imputer, _SCALERS[prep["scaler"]]()), numeric + categorical),
        ("categorical", *blocks["categorical"]),
        ("indicators", *blocks["indicators"]),
    ])
    steps = [("prep", prep_step)]
    if config["feature_selection"].get("enabled"):
        steps.append(("select", build_selector(config["feature_selection"])))
    steps.append(("model", build_model(config["model"])))
    return Pipeline(steps)


def hide_values(X, columns, n=200, seed=0):
    """Hides n known values per column; returns the data with the new gaps and the hidden truth (as in notebook 10)."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    truth = {}
    for column in columns:
        hidden = rng.choice(X.index[X[column].notna()], n, replace=False)
        truth[column] = X.loc[hidden, column].copy()
        X.loc[hidden, column] = np.nan
    return X, truth


def b2(base, X, y):
    prep = base["preprocessing"]
    numeric, categorical = prep["numeric_features"], prep["categorical_features"]
    cv = build_cv(base["cv"])
    train_idx, val_idx = next(cv.split(X, y))
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    X_va_hidden, truth = hide_values(X_va, ["age", "priors_count"])

    guesses = {}
    for name in IMPUTERS:
        imputer = _NUMERIC_IMPUTERS[name](prep.get("imputer_settings") or {}, 42)
        scaler = _SCALERS[prep["scaler"]]()
        for label, context in [("numeric only", []), ("+ categorical columns", categorical)]:
            step = ContextImputer(numeric, context, imputer, scaler).fit(X_tr) if context else None
            if step is None:
                plain = Pipeline([("scale", clone(scaler)), ("impute", clone(imputer))]).fit(X_tr[numeric])
                filled = plain["scale"].inverse_transform(plain.transform(X_va_hidden[numeric]))
            else:
                filled = step.scaler_.inverse_transform(step.transform(X_va_hidden))
            filled = pd.DataFrame(filled, index=X_va_hidden.index, columns=numeric)
            guesses.setdefault(label, {})[name] = {c: float((filled.loc[v.index, c] - v).abs().mean()) for c, v in truth.items()}
    print("### b2 guesses: done", flush=True)

    scores = {}
    for name in MODEL_IMPUTERS:
        t0 = time.time()
        folds = []
        for tr, va in cv.split(X, y):
            p = context_pipeline(base, name).fit(X.iloc[tr], y.iloc[tr])
            folds.append(float(roc_auc_score(y.iloc[va], p.predict_proba(X.iloc[va])[:, 1])))
        scores[name] = {"folds": folds, "roc_auc": [float(np.mean(folds)), float(np.std(folds, ddof=1))],
                        "seconds": round(time.time() - t0, 1)}
    print("### b2 scores: done", flush=True)
    return {"guess_mae": guesses, "scores": scores}


# ------------------------------------------------------------------------ b3
def b3(base, X, y, iterations=3):
    """
    Proximity imputation of priors_count, from scratch. A random forest is grown on the TRAINING rows of fold 1 (with
    their labels); a validation row with a hidden value is first filled roughly (training median), then routed through
    the trees; its proximity to each training row = share of trees where both land in the same leaf; the gap becomes
    the proximity-weighted average of the training rows' values. Repeat with the new value.
    """
    prep = base["preprocessing"]
    numeric, categorical = prep["numeric_features"], prep["categorical_features"]
    cv = build_cv(base["cv"])
    train_idx, val_idx = next(cv.split(X, y))
    X_tr, y_tr, X_va = X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx]
    X_va_hidden, truth = hide_values(X_va, ["age", "priors_count"])
    indicators = [f"{c}_was_missing" for c in mnar_columns(prep["imputation"])]

    fill = Pipeline([("fill", SimpleImputer(strategy="median"))]).fit(X_tr[numeric])
    codes = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]).fit(X_tr[categorical])

    def matrix(frame):
        return np.hstack([fill.transform(frame[numeric]), codes.transform(frame[categorical]), frame[indicators].to_numpy()])

    t0 = time.time()
    M_tr = matrix(X_tr)
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1).fit(M_tr, y_tr)
    leaves_tr = forest.apply(M_tr)

    out = {}
    for column in ["priors_count", "age"]:
        rows = truth[column].index
        current = X_va_hidden.loc[rows].copy()
        current[column] = np.nan
        col_known = X_tr[column].notna().to_numpy()
        col_values = X_tr[column].to_numpy()
        estimate = np.full(len(rows), np.nanmedian(col_values))
        errors = []
        for _ in range(iterations):
            current[column] = estimate
            leaves = forest.apply(matrix(current))
            proximity = (leaves[:, None, :] == leaves_tr[None, col_known, :]).mean(axis=2)     # rows x known training rows
            weights = proximity.sum(axis=1, keepdims=True)
            estimate = np.where(weights[:, 0] > 0, (proximity @ col_values[col_known]) / np.maximum(weights[:, 0], 1e-12),
                                estimate)
            errors.append(float(np.abs(estimate - truth[column].to_numpy()).mean()))
        out[column] = {"mae_per_iteration": errors}
    out["seconds"] = round(time.time() - t0, 1)
    print("### b3: done", flush=True)
    return out


def main(parts):
    os.chdir(ROOT)
    base = load_config("config.yaml")
    X, y, _ = load_development_set(base)
    out_path = ROOT / "wk10" / "imputer_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for part in parts:
        t0 = time.time()
        results[part] = {"a1": lambda: a1(base), "b1": lambda: b1(base, X, y), "b2": lambda: b2(base, X, y),
                         "b3": lambda: b3(base, X, y)}[part]()
        results[part]["_seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n### {part}: saved ({results[part]['_seconds']}s)\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["a1", "b1", "b2", "b3"])
