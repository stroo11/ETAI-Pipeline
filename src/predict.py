"""
Predicting new data -- weeks 9-10. The refit pipeline scores rows it has never seen, with no target: the same cleaning,
engineered features and feature split as the training data, but NOT drop_duplicate_rows -- every row needs a prediction.

Week 10: main.py saves the fitted pipeline (output.model_path) with the metadata from `model_metadata()`, and the
`predict.py` script next to main.py loads it to predict a CSV from the terminal, with no training.
"""
import os
import sys
from datetime import datetime

import pandas as pd

from src.data import load_data
from src.preprocessing import add_engineered_features, clean_dataset, split_features_target

# the libraries whose version decides whether a saved model still loads and behaves the same
_LIBRARIES = ["sklearn", "xgboost", "pandas", "numpy", "joblib"]


def library_versions() -> dict:
    """Installed versions of the libraries the saved pipeline depends on."""
    versions = {"python": sys.version.split()[0]}
    for name in _LIBRARIES:
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    return versions


def model_metadata(config: dict, X_train: pd.DataFrame) -> dict:
    """Saved next to the pipeline (week 10, B4): the input columns it needs, the library versions and the run's config."""
    return {"columns": list(X_train.columns), "versions": library_versions(), "config": config,
            "saved": datetime.now().isoformat(timespec="seconds")}


def prepare_features(df: pd.DataFrame, config: dict):
    """Raw rows -> (features, cleaned rows): cleaning, engineered features, feature split. Row-preserving, no target needed."""
    data_cfg, prep_cfg = config["data"], config["preprocessing"]
    clean = clean_dataset(df, prep_cfg["cleaning"])
    clean = add_engineered_features(clean, prep_cfg.get("engineered_features") or [])
    X, _, _ = split_features_target(clean, target=data_cfg["target"], sensitive_attr=data_cfg["sensitive_attr"],
                                    drop_columns=data_cfg["drop_columns"], imputation=prep_cfg["imputation"],
                                    audit_columns=data_cfg.get("audit_columns"))
    return X, clean


def predict(df: pd.DataFrame, pipeline, config: dict, threshold: float = 0.5, columns: list = None) -> pd.DataFrame:
    """
    The id, the probability of the positive class and the 0/1 prediction for EVERY input row.
    Week 10 (B4): stops with a clear message when the input lacks a column the model was trained on (`columns`, from
    the saved metadata), instead of a cryptic error from deep inside the ColumnTransformer.
    """
    X, clean = prepare_features(df, config)
    needed = columns if columns is not None else list(getattr(pipeline, "feature_names_in_", X.columns))
    lacking = [c for c in needed if c not in X.columns]
    if lacking:
        raise ValueError(f"The input lacks {len(lacking)} column(s) the model needs: {lacking}. "
                         "Check that it has the same columns as the training data.")
    probability = pipeline.predict_proba(X[needed])[:, 1]
    id_column = config["data"]["id_column"]
    out = pd.DataFrame({id_column: clean[id_column].to_numpy(), "probability": probability,
                        "prediction": (probability >= threshold).astype(int)})
    if len(out) != len(df):
        raise RuntimeError(f"{len(df)} rows in, {len(out)} predictions out: every row needs a prediction")
    return out


def predict_new_data(config: dict, model) -> str:
    """
    Week 9: if the config has a `predict` section, predicts predict.path with the final model and writes one row per
    input row to predict.output (id, probability, 0/1 prediction at predict.threshold). Returns the path, or None.
    """
    predict_cfg = config.get("predict") or {}
    if not predict_cfg.get("path"):
        return None
    out = predict(load_data(predict_cfg["path"]), model, config, threshold=predict_cfg.get("threshold", 0.5))
    path = predict_cfg["output"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return path
