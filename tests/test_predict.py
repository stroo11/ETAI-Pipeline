"""
Week 10 (B4): checks on predicting new data. Run from the project folder, with or without pytest:
    python tests/test_predict.py
    python -m pytest tests

A small, fast pipeline is fitted here (tuning off, logistic regression), so the test doesn't depend on a saved model.
"""
import copy
import os
import sys

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from main import load_config, load_development_set  # noqa: E402
from src.model import build_pipeline  # noqa: E402
from src.predict import model_metadata, predict  # noqa: E402


def fitted():
    config = load_config("config.yaml")
    config["model"] = {"type": "logistic_regression", "params": {"max_iter": 2000}}
    X_dev, y_dev, _ = load_development_set(config)
    pipeline = build_pipeline(config["preprocessing"], config["model"], config["feature_selection"]).fit(X_dev, y_dev)
    return pipeline, config, model_metadata(config, X_dev)


def test_rows_out_equal_rows_in(tmp_path=None):
    pipeline, config, meta = fitted()
    new = pd.read_csv("data/compas_new_people.csv")       # "?" placeholders, a missing age, an unseen category, a duplicate
    out = predict(new, pipeline, config, columns=meta["columns"])
    assert len(out) == len(new) == 20
    assert out["id"].tolist() == new["id"].tolist()        # same rows, same order -- duplicates kept
    assert out["probability"].between(0, 1).all() and out["prediction"].isin([0, 1]).all()


def test_saved_model_gives_the_same_predictions():
    pipeline, config, meta = fitted()
    path = os.path.join("results", "_test_model.joblib")
    os.makedirs("results", exist_ok=True)
    joblib.dump({"pipeline": pipeline, **meta}, path)
    loaded = joblib.load(path)
    os.remove(path)
    new = pd.read_csv("data/compas_new_people.csv")
    a = predict(new, pipeline, config, columns=meta["columns"])
    b = predict(new, loaded["pipeline"], config, columns=loaded["columns"])
    assert a.equals(b)


def test_missing_column_gives_a_clear_error():
    pipeline, config, meta = fitted()
    new = pd.read_csv("data/compas_new_people.csv").drop(columns=["sex"])
    try:
        predict(new, pipeline, config, columns=meta["columns"])
    except ValueError as error:
        assert "sex" in str(error) and "lacks" in str(error)
    else:
        raise AssertionError("a missing column should stop the prediction with a clear message")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
