"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean (row-preserving) -> drop duplicates (training data only)
    -> lock the test set away -> evaluate the whole pipeline on the development set (holdout or stratified k-fold CV, per config.yaml -> evaluation.method)
    -> refit the final model on all development rows -> save results
"""
import yaml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test, build_preprocessor
from src.model import build_model
from src.evaluate import (
    build_cv,
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    holdout_report,
    oof_classification_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_pipeline(prep_config: dict, model_config: dict) -> Pipeline:
    # preprocessor and model in one Pipeline, so imputation / encoding / scaling are learned from the training rows only -- in every CV fold
    return Pipeline([
        ("prep", build_preprocessor(prep_config)),
        ("model", build_model(model_config)),
    ])


def main():
    config = load_config()
    data_cfg, prep_cfg, eval_cfg = config["data"], config["preprocessing"], config["evaluation"]

    df = load_data(data_cfg["path"])
    df = clean_dataset(df, prep_cfg["cleaning"])                  # row-preserving
    df = drop_duplicate_rows(df, data_cfg.get("id_column"))        # training data only, before the split

    X, y, extras = split_features_target(
        df,
        target=data_cfg["target"],
        sensitive_attr=data_cfg["sensitive_attr"],
        drop_columns=data_cfg["drop_columns"],
        imputation=prep_cfg["imputation"],
    )
    # the locked test set is carved out here and never used again in this file
    X_dev, _X_test, y_dev, _y_test, extras_dev, _extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    pipeline = make_pipeline(prep_cfg, config["model"])

    if eval_cfg["method"] == "cv":
        cv_cfg = config["cv"]
        scoring = cv_cfg.get("scoring", "accuracy")
        fold_scores, y_oof = cross_validate_pipeline(
            pipeline, X_dev, y_dev, build_cv(cv_cfg), scoring, n_jobs=cv_cfg.get("n_jobs", 1)
        )
        report = cv_report(fold_scores, scoring)
        report += "\n" + oof_classification_report(y_dev, y_oof)
        report += "\n" + fairness_report(y_dev, y_oof, extras_dev, sensitive_attr=data_cfg["sensitive_attr"])
    elif eval_cfg["method"] == "holdout":
        X_tr, X_va, y_tr, y_va, _, extras_va = train_test_split(
            X_dev, y_dev, extras_dev,
            test_size=eval_cfg["validation_size"], random_state=eval_cfg["random_state"], stratify=y_dev,
        )
        pipeline.fit(X_tr, y_tr)
        report = holdout_report(y_tr, pipeline.predict(X_tr), y_va, pipeline.predict(X_va))
        report += "\n" + fairness_report(y_va, pipeline.predict(X_va), extras_va,
                                         sensitive_attr=data_cfg["sensitive_attr"], rows_label="validation rows")
    else:
        raise ValueError(f"Unknown evaluation method: {eval_cfg['method']}. Options: ['cv', 'holdout']")

    # evaluation scores the RECIPE; the model you'd actually use is the same recipe refit on all development rows
    final_model = make_pipeline(prep_cfg, config["model"]).fit(X_dev, y_dev)
    report += f"\nFinal model: {config['model']['type']} refit on all {len(X_dev)} development rows.\n"
    print(report.splitlines()[-1])

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")
    return final_model


if __name__ == "__main__":
    main()
