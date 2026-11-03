"""
Feature selection -- week 7 (Part B). A LEARNED step (it looks at the labels), so it lives inside the Pipeline, between "prep" and "model", and is re-fit on the training rows of every fold. Fitting it once on all development rows would leak, the week-4 way.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression


class VoteSelector(SelectorMixin, BaseEstimator):
    """Keeps the features chosen by at least `min_votes` of three models, fitted on the training rows only."""

    def __init__(self, min_votes: int = 2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=self.random_state, n_jobs=-1).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(l1_ratio=1.0, C=0.01, solver="saga", max_iter=5000).fit(X, y)
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
