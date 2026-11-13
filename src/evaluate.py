"""
Evaluation -- week 4: holdout OR stratified k-fold cross-validation of the whole pipeline, on the development set only.
Week 7: several metrics per run, computed from out-of-fold PROBABILITIES; one of them (cv.selection_metric) is what tuning maximises and runs are ranked by.

The locked test set (src/preprocessing.py:split_dev_test) is never scored here: every number this module produces may be used to compare and choose, and a score that guided a choice is no longer an unbiased estimate.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    get_scorer,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


# ------------------------------------------------------------------ splitters

def build_cv(cv_config: dict) -> StratifiedKFold:
    """The splitter from config.yaml -> cv. Same random_state = same folds on every run, for every model, so comparisons are like-for-like."""
    shuffle = cv_config.get("shuffle", True)
    return StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                           random_state=cv_config.get("random_state") if shuffle else None)


def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, ...), plus `fbeta_<beta>`
    (e.g. `fbeta_2`: recall weighted twice as much as precision).
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


# -------------------------------------------------------------------- holdout

def holdout_report(y_train, y_train_pred, y_val, y_val_pred) -> str:
    """
    Week 2/3's evaluation, now on a validation split of the development set instead of the test set: train accuracy and validation accuracy side by side (the gap is how we spot overfitting), plus the classification report on the validation rows.
    """
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    lines = [
        "Holdout (one stratified train/validation split of the development set)",
        "",
        f"Train accuracy:      {train_accuracy:.3f}",
        f"Validation accuracy: {val_accuracy:.3f}",
        f"Gap (train - validation): {train_accuracy - val_accuracy:+.3f}",
        "",
        "Classification report (validation rows):",
        classification_report(y_val, y_val_pred, zero_division=0),
    ]
    return "\n".join(lines)


# ----------------------------------------------------------- cross-validation

def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part, on every metric in `scorers`. Because `pipeline`
    contains the preprocessing too, it is re-learned inside every fold.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold with the validation score of every metric, plus the
        train score and train-validation gap of `selection_metric`;
      - oof_proba: out-of-fold probability of the positive class for every row, from the
        fold model that did not train on it. `cv` must be a partition (each row validated
        exactly once).
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X))
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table and the mean and std of every metric, as text."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """Confusion matrix and classification report of the out-of-fold predictions, as text."""
    y_pred = (oof_proba >= threshold).astype(int)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=["true 0", "true 1"],
                      columns=["predicted 0", "predicted 1"])
    return (f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
            f"{cm.to_string()}\n\n{classification_report(y_true, y_pred, zero_division=0)}")


# --------------------------------------------------------- paired comparison

def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the same CV folds.
    Folds share most of their training rows, so their scores are correlated and the plain
    t-test is overconfident; the variance of the differences is inflated by `test_train_ratio`
    (validation rows / training rows per fold: 1 / (k - 1) for k-fold).

    Returns the mean difference (a - b), the folds where a scores higher, t and the two-sided
    p-value.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}


# ------------------------------------------------------------------- fairness

def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
                    threshold: float = 0.5, rows_label: str = "development set, out-of-fold",
                    reference: dict = None, negative_label: str = "did NOT reoffend",
                    positive_label: str = "were predicted to") -> str:
    """
    False positive rate (share of the true negatives that were predicted positive) per group of
    `sensitive_attr`, for the model's predictions (probability >= `threshold`) and, if `reference`
    is given (config.yaml -> fairness.reference), for an existing score on the same rows: a row
    counts as predicted positive when `reference["column"]` is not in `reference["negative_values"]`
    (COMPAS: `score_text` other than "Low"). Week 9: the reference and the wording come from the
    config, so a dataset without COMPAS's score still gets the audit. Rows with an unknown group are
    reported as "unknown". A simple check, not a full audit.
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (np.asarray(oof_proba) >= threshold).astype(int)
    columns = [("Our model", "y_pred_model")]
    if reference:
        df["y_pred_reference"] = (~df[reference["column"]].isin(reference["negative_values"])).astype(int)
        columns.append((reference["name"], "y_pred_reference"))

    lines = [
        f"False positive rate by {sensitive_attr} ({rows_label})",
        f"(share of people who {negative_label}, but {positive_label})",
        "",
    ]
    for label, col in columns:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)
