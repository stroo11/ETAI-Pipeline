"""
Feature selection -- week 7 (Part B), week 8 (Part A). A LEARNED step (it looks at the labels), so it lives inside the Pipeline, between "prep" and "model", and is re-fit on the training rows of every fold. Fitting it once on all development rows would leak, the week-4 way.

Week 8 (Part B): the selector is a config option (feature_selection.method), so the kinds of selection can be compared on the same folds: combined (`vote`), filter (`kbest`) and embedded (`from_model`).
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectorMixin, f_classif
from sklearn.linear_model import LogisticRegression


class VoteSelector(SelectorMixin, BaseEstimator):
    """
    Keeps the features chosen by at least `min_votes` of three models, fitted on the training
    rows only: a random forest and gradient boosting vote for features with above-average
    importance, an L1-penalised logistic regression for features it doesn't shrink to zero.
    """

    def __init__(self, min_votes: int = 2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=self.random_state, n_jobs=-1).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(l1_ratio=1.0, C=0.01, solver="liblinear").fit(X, y)
        self.votes_ = ((forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
                       + (booster.feature_importances_ > booster.feature_importances_.mean())
                       + (np.abs(lasso.coef_[0]) > 1e-8))
        self.support_ = self.votes_ >= self.min_votes
        if not self.support_.any():
            # no feature reached min_votes (e.g. min_votes=3 when the lasso zeroes everything):
            # keep the most-voted ones instead of handing the model zero columns
            self.support_ = self.votes_ == self.votes_.max()
        return self

    def _get_support_mask(self):
        return self.support_


def build_selector(selection_config: dict):
    """The unfitted selector named by `selection_config["method"]`: vote (default) | kbest | from_model."""
    method = selection_config.get("method", "vote")
    seed = selection_config.get("random_state")
    if method == "vote":
        return VoteSelector(min_votes=selection_config.get("min_votes", 2), random_state=seed)
    if method == "kbest":
        return SelectKBest(f_classif, k=selection_config.get("k", 6))
    if method == "from_model":
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=seed, n_jobs=-1)
        return SelectFromModel(forest, threshold=selection_config.get("threshold", "mean"))
    raise ValueError(f"Unknown feature_selection.method: {method}. Options: ['vote', 'kbest', 'from_model']")


def kept_features(fitted_pipeline) -> list:
    """Names of the columns that reach the model of a fitted Pipeline (all of them when there is no "select" step)."""
    steps = fitted_pipeline.named_steps
    names = steps["prep"].get_feature_names_out()
    if "select" in steps:
        names = names[steps["select"].get_support()]
    return [name.split("__", 1)[-1] for name in names]
