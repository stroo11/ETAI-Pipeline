# Baseline Predictive Pipeline -- ETAI -- Name: Jovaldo Rodrigues 20260573


## WEEK 2
the best model is logistic_regression because it has a test accuracy of 0.680 (vs. 0.629 for the decision tree) and a near-zero train/test gap (-0.001, vs. +0.199 for the decision tree). The decision tree overfits: it reaches 0.829 train accuracy but drops to 0.629 on unseen data, since it splits until the leaves are pure instead of being depth-limited. Logistic regression generalizes better and is also more accurate on the test set.
