"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean (row-preserving) -> drop duplicates (training data only)
    -> lock the test set away -> evaluate the whole pipeline on the development set (holdout or stratified k-fold CV, per config.yaml -> evaluation.method)
    -> [week 5, if tuning.enabled] nested CV of the tuning procedure, then tune on all development rows
    -> refit the final (tuned) model on all development rows -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test
from src.model import build_pipeline
from src.evaluate import (
    build_cv,
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    holdout_report,
    oof_classification_report,
)
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_development_set(config: dict):
    """load -> clean -> de-duplicate -> features/target -> lock the test set. Returns (X_dev, y_dev, extras_dev)."""
    data_cfg, prep_cfg = config["data"], config["preprocessing"]

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
    # the locked test set is carved out here and never used again
    X_dev, _X_test, y_dev, _y_test, extras_dev, _extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )
    return X_dev, y_dev, extras_dev


def run(config: dict):
    """
    Runs the whole pipeline for one config and saves the report. Returns (final_model, summary), where
    summary holds the numbers behind the report (fold tables, the study, out-of-fold predictions) so
    experiments -- e.g. wk05/run_challenges.py -- can reuse this exact code path instead of copying it.
    """
    data_cfg, prep_cfg, eval_cfg = config["data"], config["preprocessing"], config["evaluation"]
    X_dev, y_dev, extras_dev = load_development_set(config)

    pipeline = build_pipeline(prep_cfg, config["model"])
    summary = {}

    if eval_cfg["method"] == "cv":
        cv_cfg = config["cv"]
        scoring = cv_cfg.get("scoring", "accuracy")
        n_jobs = cv_cfg.get("n_jobs", 1)
        cv = build_cv(cv_cfg)
        fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs=n_jobs)
        report = "Default hyperparameters (config.yaml -> model.params)\n" + cv_report(fold_scores, scoring)
        summary["default_scores"] = fold_scores

        # week 5: hyperparameter tuning -- on the development set only, every candidate scored by
        # CV of the whole pipeline (see src/tuning.py)
        tuning_config = config.get("tuning", {})
        if tuning_config.get("enabled", False):
            model_type = config["model"]["type"]
            search_spaces = tuning_config.get("search_spaces") or {}
            if model_type not in search_spaces:
                raise ValueError(f"tuning.enabled is true but config.yaml -> tuning.search_spaces has no entry for "
                                 f"'{model_type}'. Add one, or set tuning.enabled: false.")
            search_space = search_spaces[model_type]
            n_trials = tuning_config["n_trials"]
            tuning_seed = tuning_config["random_state"]
            inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

            # 1. the honest estimate (its out-of-fold predictions replace the untuned ones)
            nested_scores, y_oof = nested_cross_validate(
                pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
            )
            report += "\nNested cross-validation (outer = the folds above, inner = tuning folds)\n"
            report += cv_report(nested_scores, scoring) + "\n"
            # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
            pipeline, study = tune_pipeline(
                pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
            )
            report += tuning_report(study, nested_scores, scoring) + "\n"
            summary.update(nested_scores=nested_scores, study=study)

        summary["y_oof"] = y_oof
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

    # evaluation scores the RECIPE; the model you'd actually use is the same recipe (tuned, if tuning ran) refit on all development rows
    final_model = pipeline.fit(X_dev, y_dev)
    tuned = "tuned " if "study" in summary else ""
    report += f"\nFinal model: {tuned}{config['model']['type']} refit on all {len(X_dev)} development rows.\n"
    print(report.splitlines()[-1])

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")
    summary["results_path"] = path
    return final_model, summary


def main():
    final_model, _ = run(load_config())
    return final_model


if __name__ == "__main__":
    main()
