"""
Evaluation -- week 4: holdout OR stratified k-fold cross-validation of the whole pipeline, on the development set only.

The locked test set (src/preprocessing.py:split_dev_test) is never scored here: every number this module produces may be used to compare and choose, and a score that guided a choice is no longer an unbiased estimate.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate


# ------------------------------------------------------------------ splitters

def build_cv(cv_config: dict) -> StratifiedKFold:
    """The splitter from config.yaml -> cv. Same random_state = same folds on every run, for every model, so comparisons are like-for-like."""
    shuffle = cv_config.get("shuffle", True)
    return StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                           random_state=cv_config.get("random_state") if shuffle else None)


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
    text = "\n".join(lines)
    print(text)
    return text


# ----------------------------------------------------------- cross-validation

def cross_validate_pipeline(pipeline, X, y, cv, scoring: str = "accuracy", n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` on each fold's training part and scores it on that fold's validation part. Because `pipeline` contains the preprocessing too, imputation statistics, encoder means and scaler centres are re-learned inside every fold -- the validation fold never influences its own preprocessing.

    Returns (fold_scores, y_oof):
      - fold_scores: one row per fold -- train score, validation score, and the gap between them (a large, consistent gap = overfitting).
      - y_oof: out-of-fold predictions. Every row gets a prediction from the one fold model that did NOT train on it, so the classification report and the fairness audit run on ~5,800 unseen rows while the locked test set stays untouched.
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]

    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        y_oof[val_idx] = model.predict(X.iloc[val_idx])
    return fold_scores, y_oof


def cv_report(fold_scores: pd.DataFrame, scoring: str = "accuracy") -> str:
    """Per-fold table + mean +/- std, as text (printed, and saved to results/)."""
    lines = [
        f"Cross-validation ({len(fold_scores)} stratified folds, metric: {scoring})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""
        lines.append(f"{col.capitalize():<11s} mean = {fold_scores[col].mean():{sign}.3f}   "
                     f"std = {fold_scores[col].std(ddof=1):.3f}")
    text = "\n".join(lines) + "\n"
    print(text)
    return text


def oof_classification_report(y_true, y_pred) -> str:
    """Classification report on the out-of-fold predictions."""
    text = ("Classification report (out-of-fold predictions, development set):\n"
            + classification_report(y_true, y_pred, zero_division=0))
    print(text)
    return text


# --------------------------------------------------------- paired comparison

def paired_comparison(scores_a, scores_b, n_train: int, n_val: int) -> dict:
    """
    Is recipe A really better than recipe B, or is it fold luck? Both must have been scored on the SAME folds (same cv.random_state), so the per-fold difference cancels out how easy or hard each fold happened to be.

    The fold differences are not independent -- every pair of CV models shares most of its training rows -- so a plain paired t-test is over-confident. This is the corrected resampled t-test (Nadeau & Bengio, 2003): the variance of the mean difference is inflated from s^2/J to (1/J + n_val/n_train) * s^2.
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    J = len(diffs)
    mean, var = diffs.mean(), diffs.var(ddof=1)
    corrected_se = np.sqrt((1 / J + n_val / n_train) * var)
    t = mean / corrected_se if corrected_se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return {"mean_diff": mean, "naive_se": np.sqrt(var / J), "corrected_se": corrected_se,
            "t": t, "p_value": p, "wins": int((diffs > 0).sum()), "losses": int((diffs < 0).sum()), "folds": J}


# ------------------------------------------------------------------- fairness

def fairness_report(y_true, y_pred, extras: pd.DataFrame, sensitive_attr: str = "race",
                    rows_label: str = "development set, out-of-fold") -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show that "accurate" and "fair" are not the same thing.

    For each race group, the false positive rate (share of people who did NOT reoffend but were predicted to) for:
        - our own model (out-of-fold or validation predictions -- never the locked test set)
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk" prediction), on the same rows, for comparison
    """
    df = extras.copy()
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by race ({rows_label})",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text
