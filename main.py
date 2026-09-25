"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean -> split -> fit (preprocessor + model, on train only)
    -> evaluate (train & test) -> save results
"""
import yaml
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target, split_train_test, build_preprocessor
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    data_cfg, prep_cfg = config["data"], config["preprocessing"]

    df = load_data(data_cfg["path"])
    df = clean_dataset(df, prep_cfg["cleaning"])

    X, y, extras = split_features_target(
        df,
        target=data_cfg["target"],
        sensitive_attr=data_cfg["sensitive_attr"],
        drop_columns=data_cfg["drop_columns"],
        imputation=prep_cfg["imputation"],
    )
    X_train, X_test, y_train, y_test, extras_train, extras_test = split_train_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # preprocessor and model in one Pipeline, so imputation / encoding / scaling are learned from the training rows only
    model = Pipeline([
        ("prep", build_preprocessor(prep_cfg)),
        ("model", build_model(config["model"])),
    ])
    model.fit(X_train, y_train)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=data_cfg["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
