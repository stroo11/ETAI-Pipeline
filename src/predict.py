"""
Predicting new data -- week 9. The refit pipeline scores rows it has never seen, with no target: the same cleaning,
engineered features and feature split as the training data, but NOT drop_duplicate_rows -- every row needs a prediction.
"""
import os

import pandas as pd

from src.data import load_data
from src.preprocessing import add_engineered_features, clean_dataset, split_features_target


def predict_new_data(config: dict, model) -> str:
    """
    Reads config.yaml -> predict.path, prepares it exactly like the training data (minus de-duplication), and writes
    one row per input row to predict.output: the id, the probability of the positive class and a 0/1 prediction at
    predict.threshold. Returns the path written, or None when the config has no `predict` section.
    """
    predict_cfg = config.get("predict") or {}
    if not predict_cfg.get("path"):
        return None
    data_cfg, prep_cfg = config["data"], config["preprocessing"]
    id_column = data_cfg["id_column"]

    raw = load_data(predict_cfg["path"])
    df = clean_dataset(raw, prep_cfg["cleaning"])                                   # row-preserving
    df = add_engineered_features(df, prep_cfg.get("engineered_features") or [])
    X, _, _ = split_features_target(df, target=data_cfg["target"], sensitive_attr=data_cfg["sensitive_attr"],
                                    drop_columns=data_cfg["drop_columns"], imputation=prep_cfg["imputation"],
                                    audit_columns=data_cfg.get("audit_columns"))

    proba = model.predict_proba(X)[:, 1]
    threshold = predict_cfg.get("threshold", 0.5)
    out = pd.DataFrame({id_column: df[id_column].to_numpy(), "probability": proba,
                        "prediction": (proba >= threshold).astype(int)})
    if len(out) != len(raw):
        raise RuntimeError(f"{len(raw)} rows in, {len(out)} predictions out: every row needs a prediction")

    path = predict_cfg["output"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return path
