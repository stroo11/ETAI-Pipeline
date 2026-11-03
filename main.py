"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean (row-preserving) -> drop duplicates (training data only) -> [week 6] engineered features
    -> lock the test set away -> evaluate the whole pipeline on the development set (holdout or stratified k-fold CV, per config.yaml -> evaluation.method)
    -> [week 5, if tuning.enabled] nested CV of the tuning procedure, then tune on all development rows
    -> refit the final (tuned) model on all development rows -> save the report and [week 7] add the run to the scoreboard
"""
from datetime import datetime

import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data import load_data
from src.preprocessing import (
    add_engineered_features,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.model import build_pipeline
from src.evaluate import (
    build_cv,
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    holdout_report,
    oof_classification_report,
)
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_development_set(config: dict):
    """load -> clean -> de-duplicate -> engineered features -> features/target -> lock the test set. Returns (X_dev, y_dev, extras_dev)."""
    data_cfg, prep_cfg = config["data"], config["preprocessing"]

    df = load_data(data_cfg["path"])
    df = clean_dataset(df, prep_cfg["cleaning"])                  # row-preserving
    df = drop_duplicate_rows(df, data_cfg.get("id_column"))        # training data only, before the split
    df = add_engineered_features(df, prep_cfg.get("engineered_features") or [])   # fixed formulas: safe before the split

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
    Runs the whole pipeline for one config, saves the report and adds the run to the scoreboard.
    Returns (final_model, summary), where summary holds the numbers behind the report (fold tables,
    the study, out-of-fold probabilities) so experiments -- e.g. wk05/run_challenges.py,
    wk06/run_challenges.py -- reuse this exact code path instead of copying it.
    """
    data_cfg, prep_cfg, eval_cfg = config["data"], config["preprocessing"], config["evaluation"]
    run_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3]
    model_type = config["model"]["type"]
    X_dev, y_dev, extras_dev = load_development_set(config)

    def make_pipeline(model_config):
        return build_pipeline(prep_cfg, model_config, config.get("feature_selection"))

    pipeline = make_pipeline(config["model"])
    summary = {"run": run_id}

    if eval_cfg["method"] == "holdout":
        X_tr, X_va, y_tr, y_va, _, extras_va = train_test_split(
            X_dev, y_dev, extras_dev,
            test_size=eval_cfg["validation_size"], random_state=eval_cfg["random_state"], stratify=y_dev,
        )
        pipeline.fit(X_tr, y_tr)
        report = holdout_report(y_tr, pipeline.predict(X_tr), y_va, pipeline.predict(X_va))
        report += "\n" + fairness_report(y_va, pipeline.predict_proba(X_va)[:, 1], extras_va,
                                         sensitive_attr=data_cfg["sensitive_attr"], rows_label="validation rows")
        print(report)
        path = save_run(config.get("output", {}).get("results_dir", "results"), run_id, config, report)
        print(f"Full results saved to {path}")
        return pipeline.fit(X_dev, y_dev), summary
    if eval_cfg["method"] != "cv":
        raise ValueError(f"Unknown evaluation method: {eval_cfg['method']}. Options: ['cv', 'holdout']")

    cv_cfg = config["cv"]
    cv = build_cv(cv_cfg)
    metrics, selection_metric = cv_cfg["metrics"], cv_cfg["selection_metric"]
    if selection_metric not in metrics:
        raise ValueError(f"cv.selection_metric '{selection_metric}' must be one of cv.metrics: {metrics}")
    scorers = build_scorers(metrics)
    n_jobs = cv_cfg.get("n_jobs", 1)
    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   {cv_cfg['n_splits']}-fold CV")
    print(f"Data:  {len(X_dev)} development rows (the locked test set is not used)")

    # week 7: the no-model floor (a dummy) on the same folds, so every run shows what the model adds
    floor_scores, _ = cross_validate_pipeline(make_pipeline({"type": "dummy"}), X_dev, y_dev, cv,
                                              {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
    floor, untuned = floor_scores["validation"].mean(), fold_scores["validation"].mean()
    summary.update(default_scores=fold_scores, floor=floor)

    print("\nDefault hyperparameters (config.yaml)")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
    report = (f"Hyperparameters from config.yaml: {config['model'].get('params')}\n\n"
              + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
              + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
    final_scores = fold_scores

    # week 5: hyperparameter tuning -- on the development set only, every candidate scored by
    # CV of the whole pipeline (see src/tuning.py)
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    study = None
    if tuning_enabled:
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning.enabled is true but config.yaml -> tuning.search_spaces has no entry for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = dict(search_spaces[model_type])
        selection_cfg = config.get("feature_selection") or {}
        if selection_cfg.get("enabled", False) and selection_cfg.get("search_space"):
            search_space.update(selection_cfg["search_space"])    # week 7: tune select__min_votes with the model
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # 1. the honest estimate on the same outer folds; its out-of-fold probabilities feed the reports
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
        report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                     "Nested cross-validation (the tuning procedure, estimated honestly)")

        # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
        pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, tuning_seed, n_jobs=n_jobs)
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            print(f"    {name.replace('model__', ''):<28s} {value:.4g}" if isinstance(value, float)
                  else f"    {name.replace('model__', ''):<28s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)
        summary.update(nested_scores=final_scores, study=study)

    print(f"\nAll metrics ({'tuned model, nested CV' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
    summary["oof_proba"] = oof_proba
    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=data_cfg["sensitive_attr"])

    # evaluation scores the RECIPE; the model you'd actually use is the same recipe (tuned, if tuning ran) refit on all development rows
    final_model = pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {'tuned ' if tuning_enabled else ''}{model_type} refit on all {len(X_dev)} development rows."
    report += "\n" + refit + "\n"

    # week 7: the scoreboard -- one row per run, comparable with every run on the same data and folds
    selection_cfg = config.get("feature_selection") or {}
    row = {"run": run_id, "data": data_cfg["path"], "model": model_type, "tuned": tuning_enabled,
           "encoder": prep_cfg["encoder"], "scaler": prep_cfg["scaler"],
           "features": "+".join(prep_cfg.get("engineered_features") or []) or "base",
           "selection": bool(selection_cfg.get("enabled", False)),
           "params": str(config["model"].get("params")), "tuned_params": str(study.best_params) if study else "",
           "metric": selection_metric, "mean": final_scores["validation"].mean(),
           "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}}
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
    board = (same.sort_values("mean", ascending=False)
             [["run", "model", "tuned", "features", "selection", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    path = save_run(config.get("output", {}).get("results_dir", "results"), run_id, config, report)
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")
    summary["results_path"] = path
    return final_model, summary


def main():
    final_model, _ = run(load_config())
    return final_model


if __name__ == "__main__":
    main()
