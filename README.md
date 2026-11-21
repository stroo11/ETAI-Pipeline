# Predictive Pipeline -- ETAI -- Name: Jovaldo Rodrigues 20260573

This project predicts whether a defendant reoffends within two years (`two_year_recid`) using ProPublica's COMPAS dataset. Race is never used as a model input. It is kept aside only to check whether the model makes more mistakes for some groups than for others, and to compare our model with COMPAS's own risk score.

Run it with `python main.py`. Everything tunable lives in `config.yaml`.

## Pipeline progress

| Week | What changed | How it is evaluated | Best model (score) |
|---|---|---|---|
| 2 | Baseline: drop missing rows, one-hot everything | one 80/20 split | Logistic regression, 0.680 (test) |
| 3 | EDA-driven cleaning, imputation by missingness mechanism, leak-safe `Pipeline` | one 80/20 split (+ 15 repeated splits) | Decision tree, depth 5, 0.665 (test) |
| 4 | Row-preserving cleaning, locked test set, cross-validation, dummy + random forest, configurable imputer (KNN) and log1p | stratified 5-fold CV on the development set | Random forest, `min_samples_leaf=40`, **0.683 ± 0.016** (CV) |
| 5 | Hyperparameter tuning with Optuna (TPE) for the tree, logistic regression and random forest; `build_pipeline()`; search spaces in `config.yaml` | nested CV: 5 outer folds (same as week 4) × Optuna with 5 inner folds | Tuned random forest, **0.678 ± 0.012** (nested CV) |
| 6 | Ensembles: oblique random forest and AdaBoost (`oblique_random_forest()`, `adaboost()`); random forest with fixed 200 trees; engineered features (`add_engineered_features()`, off by default) | nested CV (accuracy), same folds | Tuned AdaBoost, **0.684 ± 0.014** (nested CV), tied with both forests |
| 7 | Gradient boosting, HistGradientBoosting and Naive Bayes; 8 metrics from out-of-fold probabilities, ROC-AUC as the selection metric; the scoreboard (`results/runs.csv`); optional learned feature selection (`VoteSelector`) | nested CV, **ROC-AUC**, same folds | Tuned HistGradientBoosting / gradient boosting, **0.728 ± 0.016** ROC-AUC (nested CV), tied with AdaBoost and the oblique forest |
| 8 | XGBoost; feature selection as a Pipeline step, on by default (`vote`, `kbest`, `from_model`); cached preprocessing and selection | nested CV, ROC-AUC, same folds | Tuned XGBoost + `vote` selection, **0.726 ± 0.013** ROC-AUC (nested CV) on 2 features, tied with the larger boosters (0.728) |
| 9 | Option A, beyond COMPAS: the same pipeline on the Adult income data (`config_adult.yaml`); cleaning, fairness and the run made general; predictions for new data (`src/predict.py`) | nested CV, ROC-AUC, on Adult's own development set | Adult: tuned XGBoost + `vote` selection, **0.910 ± 0.014** ROC-AUC, 0.857 accuracy, 0.668 F1 (all three targets met). COMPAS unchanged |
| 10 | The numeric imputer as a setting (median, mean, KNN, MICE, missForest; numeric columns scaled first); the final pipeline saved to `models/`; `predict.py` predicts a CSV from the terminal | nested CV, ROC-AUC, same folds | Unchanged: tuned XGBoost + `vote`, **0.726 ± 0.013** (COMPAS), **0.910 ± 0.014** (Adult). The imputer makes no detectable difference, so `median` stays |

## Week 2 -- Baseline

**What was done:** I built a first, deliberately simple pipeline. Rows with any missing value were dropped, every text column was one-hot encoded, and the data was split once into 80% train and 20% test. I trained two models on it, a logistic regression and a decision tree, and compared their accuracy on the training and test sets.

**What was concluded:** Logistic regression was the better model, with 0.680 test accuracy and almost no gap between train and test (-0.001). The decision tree overfitted: it reached 0.829 on the training data but dropped to 0.629 on unseen data, because without a depth limit it keeps splitting until it memorizes the training rows. Limiting the tree to a depth of 5 fixed most of this (0.680 train, 0.668 test), but logistic regression was still slightly ahead.

## Week 3 -- Data cleaning and leak-safe preprocessing

**What was done:** I replaced the naive preprocessing with a proper cleaning pipeline based on the EDA diagnosis. The same category written in different ways (for example `Male`, `MALE` and ` male`) is now merged into one label, placeholders like `?` and `-` are treated as missing values, and impossible values (negative ages, negative counts, more than 60 priors, COMPAS scores above 10) are turned into missing values too. I also removed 72 duplicate rows and three redundant columns that only repeated information from other columns.

Missing values are no longer dropped. They are filled in, and the method depends on why the value is missing. When it is missing at random, I simply use the median or the most frequent value. For `priors_count` and `c_charge_degree` the fact that the value is missing seems to carry information, so besides filling it I add a flag column that tells the model it was missing. All of these steps learn only from the training data, so no information from the test set leaks into the model, and the pipeline can also run on new data without the target column.

While checking the data I found two problems the EDA had not covered. First, 217 missing values in `juv_fel_count` could be recovered exactly, because `juvenile_total` is always the sum of the three juvenile counts. Second, 6 rows had an age group that contradicted the actual age (for example, a 51-year-old labelled "Less than 25"), so I now rebuild the age group from the age. Finally, I tested 12 combinations of category encoder and scaler. Target encoding with min-max scaling came out first, but all the combinations were within 0.003 of each other, so this choice makes very little difference.

**What was concluded:** Cleaning the data did not make the models much more accurate, but it made the results much more trustworthy. Because a single train/test split can be lucky or unlucky, I compared the models over 15 different splits:

| Model | Week 2 | Week 3 |
|---|---|---|
| Logistic regression | 0.671 | 0.672 |
| Decision tree (no depth limit) | 0.617 | 0.603 |
| Decision tree (max depth 5) | 0.662 | **0.675** |

Logistic regression stayed the same. On the original single split it even seemed to get worse (0.680 → 0.658), but that drop is mostly down to which rows happened to land in the test set. The depth-limited tree was the only model that clearly improved. The unlimited tree still overfits and is still the worst model.

The biggest gains were in correctness. The week 2 pipeline threw away 14% of the people because of missing values, and it read `priors_count` as text, so every number of prior offenses became its own unrelated category (40 of them). The fairness check was broken too: race had 17 different spellings, so each group was counted only in part, and COMPAS scores written as "low" or "LOW" were wrongly counted as high risk. The week 3 numbers are the first ones I can rely on.

On fairness, the rate of people wrongly predicted to reoffend went down for every group. However, the gap between African-American and Caucasian defendants did not shrink and actually widened a little (0.10 → 0.14 with logistic regression), even though race is not a model input. Our models still make far fewer of these errors than COMPAS itself, which wrongly flags 44% of African-American and 24% of Caucasian defendants who did not reoffend.

**Best model at the end of week 3:** the decision tree with a maximum depth of 5, with 0.675 average test accuracy and a small gap between train and test. It beat logistic regression on 10 of the 15 splits, but only by 0.003 on average, so in practice the two models are tied. Logistic regression remains a good alternative that is also easier to interpret. *(Week 4 confirmed that the two are tied, see below.)*

## Week 4 -- Preprocessing in the pipeline, cross-validation, and a better recipe

**What was done:**

- **The pipeline now has a locked test set.** `split_dev_test()` sets 20% of the data aside (stratified, seed 42). Nothing is fitted, compared or chosen on it. Everything else, the *development set* of 5,771 people, is where models are evaluated.
- **Cleaning keeps every row.** `clean_dataset()` no longer drops duplicates, because the same function has to work on new data where every row needs a prediction. Removing the 72 duplicates is now a separate, training-only step (`drop_duplicate_rows()`) that runs before the split, so the same person can't end up in both the training and the test data.
- **Cross-validation.** `main.py` evaluates the *whole* pipeline (preprocessing + model) with stratified 5-fold cross-validation (`config.yaml` → `evaluation.method: cv`). Because the preprocessing sits inside the pipeline, the medians, target-encoding means and scaler values are re-learned in every fold. The classification report and the fairness check use the out-of-fold predictions, so every one of the 5,771 people gets a prediction from a model that never saw them. The old single split is still available (`evaluation.method: holdout`).
- **New models:** `dummy`, which always predicts the most common outcome and is the floor every model must beat, and `random_forest`.
- **New preprocessing options:** a KNN imputer (`numeric_imputer: knn`, which scales before imputing because KNN uses distances), mean imputation, and a `log1p` transform for skewed counts (`log_features`).
- **A fair way to compare two recipes:** `paired_comparison()` scores both recipes on the same folds and uses a corrected t-test (Nadeau & Bengio), because CV folds share most of their training rows and a normal t-test is too confident.

## Week 5 -- Hyperparameter tuning

**What was done:**

- **Tuning is part of the pipeline.** With `tuning.enabled: true`, `main.py` tunes the configured model with Optuna (`src/tuning.py`) and refits the tuned pipeline on all development rows. With `false`, it only cross-validates `model.params`, which are now the plain defaults (the week-4 `min_samples_leaf=40` is no longer hard-coded: the tuner finds it).
- **One place builds the estimator.** `build_pipeline()` (`src/model.py`) puts preprocessing and model into one `Pipeline`; CV, tuning and the final refit all use it. Its step names address hyperparameters from outside (`model__max_depth`, `prep__numeric__impute__strategy`).
- **Search spaces live in `config.yaml`**, one per model (`tuning.search_spaces`): the tree (`max_depth`, `min_samples_leaf`, `criterion`), logistic regression (`C`, log scale from 0.0001 to 100) and the random forest (`n_estimators`, `max_depth` incl. `None`, `min_samples_leaf`, `max_features`).
- **Nested cross-validation** (`nested_cross_validate()`) gives the honest score for "tune, then refit". Its out-of-fold predictions feed the classification report and the fairness audit. `tuning_report()` prints the top trials and the optimism check.
- `main.run(config)` runs the whole pipeline for one config, so `wk05/run_challenges.py` reuses the exact `python main.py` code path for every challenge scenario (it writes the numbers to `wk05/challenge_results.json`, a generated file that is git-ignored).

## Week 6 -- Ensembles: random forest, oblique random forest and AdaBoost

**What was done:**

- **Two new model types** in `src/model.py`. `oblique_random_forest()` builds a forest whose splits are random linear combinations of features (a `SparseRandomProjection` followed by a decision tree, bagged). `adaboost()` is AdaBoost over decision trees of a chosen depth. Both are registered in `_MODELS`, so `model.type` can name them. They are functions rather than classes, because the base tree is an object and `config.yaml` can only hold plain values.
- **Search spaces in `config.yaml`** follow from how each ensemble overfits. The random forest keeps `n_estimators` fixed at 200 (more trees never overfit, they only cost time) and tunes tree size and decorrelation (`max_depth`, `min_samples_leaf`, `max_features`). The oblique forest tunes the same tree size through the nested path `model__estimator__tree__...`, plus `n_components`. AdaBoost tunes how weak its base trees are (`model__estimator__max_depth`) together with `n_estimators` and `learning_rate`, because those two trade off.
- **Feature engineering** (`add_engineered_features()` in `src/preprocessing.py`, switched on with `preprocessing.engineered_features`). It offers the course's `priors_per_adult_year` and three features of my own (challenge 1 below). All four are fixed formulas, so they are computed before the split and join the numeric block.
- `wk06/run_challenges.py` runs every scenario through `main.run()` (results in the git-ignored `wk06/challenge_results.json`).

## Week 7 -- Boosting, metrics from probabilities, and the scoreboard

**What was done:**

- **Three new model types** in `_MODELS`: `gradient_boosting` (`GradientBoostingClassifier`: regression trees fit to the gradient of the log-loss, summed in log-odds), `hist_gradient_boosting` (the same algorithm on binned features: faster, with native missing values), and `naive_bayes` (`GaussianNB`, a fast probabilistic baseline). Each has a search space in `config.yaml`. For the boosters, the number of trees, the learning rate and the tree size are searched **together**, because in boosting more trees *can* overfit. Gradient boosting is now the configured model.
- **Several metrics, from probabilities.** `cross_validate_pipeline()` now returns out-of-fold **probabilities** instead of labels and scores every metric in `cv.metrics` (accuracy, balanced accuracy, precision, recall, F1, F2, ROC-AUC, average precision). `build_scorers()` builds the scorers, including `fbeta_<beta>`. The classification report (now with a confusion matrix) and the fairness check take the probabilities plus a threshold (0.5), so the threshold stays a separate, visible decision.
- **One selection metric** (`cv.selection_metric: roc_auc`) is what Optuna maximizes and what runs are ranked by. `tune_pipeline()` takes its scorer, and `nested_cross_validate()` scores every metric on each outer fold.
- **The scoreboard.** Every run adds one row to `results/runs.csv` (`save_scores()`): model, settings, engineered features, selection on/off, the selection metric's mean, std and per-fold scores, the **untuned** score of the same model and the **no-model floor** (a dummy on the same folds). `main.py` prints the board, best first, for runs on the same data and metric. Rows are paired by fold, so any two can be compared with `corrected_paired_test()`, which replaces week 4's `paired_comparison()` (same Nadeau & Bengio test, with the course's interface).
- **Learned feature selection (Part B)** is an optional `"select"` step between `"prep"` and `"model"` (`src/selection.py`, `config.yaml` → `feature_selection`). A random forest, gradient boosting and an L1 logistic regression vote, and a feature is kept with at least `min_votes` votes. `select__min_votes` is tuned with the model. I found one edge case and handled it: with `min_votes=3` it could keep **zero** features (the L1 model zeroes everything) and crash the model. It now keeps the most-voted features instead.
- `wk07/run_scoreboard.py` runs every model through `main.run()`, tuned where it has a search space (results also in the git-ignored `wk07/scoreboard_results.json`).

## Week 8 -- XGBoost and feature selection, done right

**What was done:**

- **XGBoost** (`"xgboost"` in `_MODELS`, `xgboost>=2.0` in `requirements.txt`) is gradient boosting with brakes inside every split (`gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`). It is now the configured model, and its search space tunes those brakes together with the number of trees, the learning rate and the depth.
- **Feature selection in the pipeline.** `build_pipeline()` adds a `"select"` step between `"prep"` and `"model"` when `feature_selection.enabled` is on, so selection is re-fit on the training rows of every fold. `select__min_votes` joins the model's search space, so the reported score is nested CV. The output and the report end with the features the final model keeps, and the scoreboard has a `feature_selection` column (I renamed the `selection` column from week 7 in the existing rows).
- **Caching.** `main.py` gives each run a temporary cache folder (`memory=` in the `Pipeline`, deleted at the end). Fitted `prep` and `select` steps are reused when a later trial uses the same rows and settings, so tuning the model doesn't refit the three voters every time. It changes the cost, not the scores.
- **Other selectors (Part B).** `feature_selection.method` chooses `vote`, `kbest` (ANOVA F-test filter) or `from_model` (random forest importances), each with its own entry under `feature_selection.search_space`.
- `VoteSelector` lives in `src/selection.py` rather than `src/model.py`, with the course's `liblinear` L1 voter. It keeps my week 7 guard for when no feature reaches `min_votes`.
- `w08/run_challenges.py` runs every scenario through `main.run()` (results in the git-ignored `w08/challenge_results.json`).

## Week 10 -- Missing values and deployment

**What was done:**

- **The numeric imputer is a setting** (`preprocessing.numeric_imputer`). The options are `median`, `mean`, `knn`, `iterative` (MICE with one completed dataset) and `random_forest` (the missForest idea: `IterativeImputer` with a random forest per column). They are registered in `_NUMERIC_IMPUTERS` (`src/preprocessing.py`), like the scalers and encoders, and their options live in `preprocessing.imputer_settings` (`n_neighbors`, `max_iter`, `n_estimators`). All of them learn from the **feature** columns only, never from the target, so they run unchanged inside cross-validation and on new people. My config keeps its week 4 names (`numeric_imputer`) rather than the course's `imputation.numeric_strategy`, because `imputation` already holds my per-column missingness plan.
- **The numeric columns are scaled first, then imputed**, for every imputer. KNN and the model-based imputers compare columns, so the columns must share one scale, and scikit-learn's scalers ignore gaps when they learn it. With `median` and the min-max scaler the order changes nothing: the COMPAS and Adult reports came out **identical, character for character**, to the ones from before today.
- **The final pipeline is saved.** At the end of a run, `main.py` writes the refit pipeline to `output.model_path` (`models/model.joblib`, or `models/adult_model.joblib` for Adult) with `joblib`. It saves it together with the input columns it needs, the library versions and the run's config. The run's cache folder is detached first (`memory=None`).
- **`predict.py`** (next to `main.py`) loads that file and predicts a CSV from the terminal, with no training. The functions it uses live in `src/predict.py` (`prepare_features()`, `predict()`), and `main.py` uses the same ones for Adult's predictions file, so there is one prediction path.
- **A safer `predict.py` (B4):** it stops with a clear message when the input lacks a column the model needs, warns when a library version differs from the one the model was saved with, and refuses to write a file with fewer rows than it read. `tests/test_predict.py` checks that rows out = rows in (duplicates and gaps included), that a saved and reloaded model predicts the same, and that a missing column gives the clear error (`python tests/test_predict.py`).
- `models/` and `predictions.csv` are git-ignored. `joblib` is in `requirements.txt`, and `data/compas_new_people.csv` (20 new people) is in `data/`. `wk10/run_imputers.py` reruns every comparison below.

## Model evaluation

The test set is locked. Models are compared with stratified 5-fold cross-validation on the development set, and the same seed gives the same folds for every model.

| Model | Holdout accuracy (W3) | CV accuracy (mean ± std) | CV train–val gap |
|---|---|---|---|
| Dummy | — | 0.549 ± 0.000 | -0.000 |
| Logistic regression | 0.658 | 0.672 ± 0.015 | +0.001 |
| Decision tree (no depth limit) | — | 0.614 ± 0.013 | +0.082 |
| Decision tree (max depth 5) | 0.665 | 0.674 ± 0.017 | +0.010 |
| Random forest (default) | — | 0.649 ± 0.018 | +0.083 |
| **Random forest (`min_samples_leaf=40`)** | — | **0.683 ± 0.016** | **+0.005** |

**Which number I trust, and why:** the cross-validation mean, reported together with its spread. When I scored the same pipeline on 30 different 75/25 splits, logistic regression ranged from 0.656 to 0.699, only because of which rows ended up in the validation set. That spread is bigger than almost any improvement I am trying to measure. Repeated 20 times with different seeds, the 5-fold estimate moved 5 to 7 times less than a single holdout split.

**Goal 1 -- week 3 vs week 4 preprocessing (holdout):** on the same train/validation split, the week 3 and week 4 recipes produce *exactly* the same validation matrix, so logistic regression (0.678) and the depth-5 tree (0.678) get identical scores. Week 3 was already leak-safe, so this week's changes are about structure: data to predict keeps all its rows, and the test set stays locked. The unlimited tree and the default forest did move (0.631 → 0.596), but only because the target encoder's internal seed changed. That is a sign of how unstable they are, not an improvement or a regression. The jump compared with the recorded week 3 numbers (0.658 / 0.665) is not an improvement either: those were measured on different rows (the now-locked test set).

**Does the week 3 "best model" still hold? No.** Compared on the same 25 folds, the depth-5 tree beats logistic regression by +0.01 percentage points, winning 13 folds and losing 12 (corrected p = 0.99). The two are tied, and the week 3 ranking was noise.

**Goal 3 -- changing the pipeline (5×5 repeated CV, same folds, paired corrected test):**

| Change (vs median + target + min-max) | Logistic regression | Tree, depth 5 | Random forest (default) |
|---|---|---|---|
| KNN imputer (k=5) | -0.21 pp (p=0.08) | -0.18 pp | +0.11 pp |
| Mean imputer | -0.07 pp | -0.26 pp | -0.26 pp |
| log1p on counts | +0.33 pp (p=0.55) | 0 (identical) | +0.04 pp |
| One-hot encoder | -0.47 pp | +0.10 pp | **-1.55 pp (p=0.04)** |
| Standard / robust / no scaler | ≈ +0.1 pp | 0 (identical) | ≈ 0 |

**No preprocessing change improved the pipeline.** The KNN imputer is slightly *worse*, for three reasons. Very little is missing (6.9% of `priors_count`). The missingness is informative, and the `_was_missing` flag already captures that: people with a missing `priors_count` reoffend 35% of the time, against 46% for everyone else. And the only columns KNN can use to find similar people (age and juvenile counts, which are almost always 0) say little about prior offenses, so the filled-in values are mostly noise. Trees ignore `log1p` and scaling completely, because a split like "priors > 3" divides the people the same way under any order-preserving transformation. One-hot encoding looked much worse for the default forest, but that turned out to be a side effect. During training, the target encoder gives the same category slightly different values in different internal folds, and this noise was accidentally stopping an overfitting forest from memorizing the training rows. Once the forest is properly regularized, all encoders score the same (0.683 / 0.678 / 0.682).

**What did help: regularizing the random forest.** Requiring at least 40 people in every leaf stops each tree from memorizing individuals. This raised CV accuracy from 0.649 to 0.683 and closed the train–validation gap from +0.083 to +0.005. Because I picked this setting from 17 candidates, I re-checked it on 25 *new* folds that played no part in the choice. It still beat the week 3 tree by +0.76 percentage points, winning 21 of 24 folds (corrected p = 0.08). That is a small but consistent improvement, and it was the model in `config.yaml` in week 4 (week 5 replaces it with the tuned forest).

**Fairness (out-of-fold, development set):** our models still make far fewer false accusations than COMPAS, which wrongly flags 45% of African-American and 23% of Caucasian defendants who did not reoffend, against 26% / 13% for logistic regression and 30% / 16% for the regularized forest. The gap between the two groups (about 0.13) is the same in every model and has not shrunk since week 3. The tuned forest predicts "reoffends" more often, so it catches more people who do reoffend (recall 0.569 vs 0.511), but its false positive rate is 1 to 4 points higher in every group. This is the cost of its higher accuracy, and the next step is to choose the decision threshold based on what each type of error costs.

**Best model at the end of week 4:** a random forest with 300 trees and `min_samples_leaf=40`: **0.683 ± 0.016** CV accuracy, train–validation gap +0.005, and roughly 1 point better than the week 3 tree, confirmed on fresh folds. The depth-5 tree (0.674) and logistic regression (0.672) remain close, and they are simpler to explain.

### Week 5 -- Hyperparameter tuning

**How the pipeline tunes.** Optuna's TPE sampler (seeded with `tuning.random_state: 42`) proposes hyperparameters from the model's search space in `config.yaml`. Every trial clones the *whole* pipeline, sets the candidate values, and scores it with 5-fold CV on the development set (`tuning.n_splits: 5`). The inner folds are the same for every trial, so candidates are compared like-for-like. I ran 30 trials for the tree and logistic regression and 15 for the forest, with `n_estimators` kept between 50 and 300. **Why it doesn't leak:** the test set is still locked. The imputer, encoder and scaler are re-fit on the training part of every fold of every trial. And the score I report is not the best trial's score but **nested CV**: the tuning is repeated from scratch inside each of the 5 outer folds (the same folds as week 4), and the tuned model is scored on the outer-validation rows, which played no part in choosing it.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.614 ± 0.013 | 0.676 ± 0.019 | 0.680 | -0.001 | `max_depth=6`, `min_samples_leaf=110`, `criterion=entropy` |
| Logistic regression | 0.672 ± 0.015 | 0.671 ± 0.013 | 0.674 | +0.001 | `C=4.55` |
| **Random forest** | 0.646 ± 0.018 | **0.678 ± 0.012** | 0.681 | +0.004 | `n_estimators=128`, `max_depth=8`, `min_samples_leaf=36`, `max_features=0.5` |

*Default = scikit-learn's defaults (LR with `max_iter=2000`; the forest with 100 trees, which is why it scores 0.646 here and 0.649 in the week-4 table, where it had 300). Optimism = mean tuning score inside the outer folds (`inner_best`) minus the mean nested CV score. Chosen hyperparameters = one tuning run on the whole development set.*

The tree and the forest gained a lot from tuning: +6.2 and +3.2 points, about 3 to 5 times the fold-to-fold std for the tree and 2 to 3 times for the forest. Both gains come from the same thing. Without limits, the trees memorize the training rows (train–validation gap +0.08), and tuning regularizes them (gap ≈ +0.01). Logistic regression gained nothing (-0.001, well inside its ± 0.013), because it was never overfitting in the first place. Tuning changed the ranking: the tree went from worst to level with the others. But the three tuned models are now within 0.007 of each other, less than one std, so in practice they are tied. The forest is only nominally best. For the forest I would report **0.678 ± 0.012**, the nested CV score. It measures the whole procedure (tune with 15 trials, then refit) on rows that took no part in the choice. The best-trial score (0.681), and week 4's hand-picked 0.683, were both chosen as the maximum of several noisy scores, so they are slightly optimistic. Here the optimism is small (≤ 0.004) because the search spaces are small and 5 folds average out a lot of the noise.

**Challenge 1 -- tuning logistic regression.** `C` searched on a log scale from 0.0001 to 100 barely matters. Every value from about 0.1 to 100 scores between 0.668 and 0.672 (5-fold CV), and the five outer folds picked anything from `C=0.19` to `C=19`. Only very strong regularization hurts: `C=0.01` drops to 0.584 and `C=0.0001` to the dummy's 0.549. A linear model with about 10 coefficients and about 4,600 training rows per fold has very little variance to remove. The default (`C=1`) is already in the flat region, so regularization has nothing left to fix. The tree has thousands of possible leaves, so its capacity, and with it its overfitting, depends almost entirely on its hyperparameters.

**Challenge 2 -- tuning the random forest.** The forest prefers **deeper trees than the single tree** (`max_depth=8` vs 6, with three of the five outer folds also choosing 8). It also chose `min_samples_leaf=36`, almost exactly the 40 I picked by hand in week 4. That fits the bias–variance picture. A single tree has to stay shallow, because depth adds variance. A forest averages many decorrelated trees (`max_features=0.5`), which cancels much of that variance, so each tree can afford to be deeper and less biased. In both models the real brake is the minimum leaf size, not the depth.

**Challenge 3 -- tuning the preprocessing too.** Adding `prep__numeric__impute__strategy: [median, mean]` to the tree's space: the tuner kept `median` (on the full development set and in 4 of 5 outer folds), and nested CV was 0.673 ± 0.012, inside the noise of the 0.676 above. This confirms week 4's finding that the imputer barely matters here. It is still **leak-free**, because the strategy is just another `Pipeline` parameter: the imputer is re-fit on the training part of each fold of each trial. It would **not** be valid with preprocessing cached once outside the trials. A cache stores folds that were already transformed with *one* strategy, so the "other" strategy would never really be tried. And fitting the preprocessing once on all development rows would let validation rows shape their own features (week 5 mistake 2).

**Challenge 4 -- `cv.scoring: balanced_accuracy`** (out-of-fold, nested CV):

| Model, tuned for | Chosen hyperparameters | Accuracy | Balanced acc. | Recall (reoffends) | FPR African-American | FPR Caucasian |
|---|---|---|---|---|---|---|
| Tree, accuracy | depth 6, leaf 110, entropy | 0.676 | 0.665 | 0.552 | 0.29 | 0.17 |
| Tree, balanced acc. | depth 13, leaf 111, gini | 0.671 | 0.663 | 0.583 | 0.33 | 0.19 |
| LR, accuracy | C = 4.55 | 0.671 | 0.656 | 0.502 | 0.26 | 0.13 |
| LR, balanced acc. | C = 4.17 | 0.673 | 0.659 | 0.517 | 0.27 | 0.14 |
| Forest, accuracy | 128 trees, depth 8, leaf 36, 0.5 | 0.678 | 0.667 | 0.559 | 0.30 | 0.16 |
| Forest, balanced acc. | 203 trees, depth 20, leaf 35, sqrt | 0.679 | 0.669 | 0.568 | 0.30 | 0.17 |

The chosen hyperparameters change for the tree (depth 6 → 13) and the forest (depth 8 → 20, `sqrt`), but hardly for LR. Recall of the positive class goes up a little in every model (+0.8 points for the forest, +1.5 for LR, +3.1 for the tree), because the models predict "reoffends" more often. The price is paid in false positives, and it is paid in every group: the African-American FPR rises from 0.29 to 0.33 for the tree, and the gap to Caucasian defendants (about 0.13) does not shrink. The effect is small because the classes are only mildly imbalanced (45% reoffend) and the models still predict with a 0.5 threshold. Changing the metric changes which hyperparameters win, not the decision rule. Choosing the threshold, or `class_weight`, is the stronger lever here, and it should be chosen based on what each kind of error costs.

**Current best model:** the tuned random forest (`n_estimators=128`, `max_depth=8`, `min_samples_leaf=36`, `max_features=0.5`), **0.678 ± 0.012** nested CV accuracy. It is tied with the tuned tree (0.676) and logistic regression (0.671). It is the model in `config.yaml`, and `python main.py` re-tunes and refits it.

### Week 6 -- Ensembles

Accuracy, nested CV, the same 5 outer folds as every week (30 Optuna trials, 5 inner folds). Gap = train − validation accuracy.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.015 (+0.001) | 0.671 ± 0.013 (+0.002) | 0.674 | `C=4.55` |
| Decision tree | 0.614 ± 0.013 (+0.082) | 0.676 ± 0.019 (+0.009) | 0.680 | `max_depth=6`, `min_samples_leaf=110`, `criterion=entropy` |
| Random forest | 0.646 ± 0.018 (+0.086) | 0.681 ± 0.015 (+0.008) | 0.684 | `max_depth=24`, `min_samples_leaf=45`, `max_features=0.5` (200 trees) |
| Oblique random forest | 0.647 ± 0.016 (+0.110) | 0.680 ± 0.012 (+0.010) | 0.681 | tree `max_depth=15`, `min_samples_leaf=55`, `n_components=4` (200 trees) |
| **AdaBoost** | 0.676 ± 0.013 (+0.006) | **0.684 ± 0.014 (+0.004)** | 0.683 | base tree `max_depth=4`, `n_estimators=220`, `learning_rate=0.025` |

AdaBoost has the best mean, but its lead is not real. It is +0.003 over the random forest (4 of 5 folds, corrected p = 0.47), +0.004 over the oblique forest (p = 0.54) and +0.013 over logistic regression (5 of 5 folds, but p = 0.19). Every lead is smaller than the fold-to-fold std (≈ 0.014). The tuned ensembles are tied, and they are only about one point above logistic regression. The chosen hyperparameters do match how each model works. Both forests pick **deep** trees (24 and 15) braked by **large leaves** (45 and 55 people): averaging removes variance, so the trees can be complex, but not so complex that they memorize individuals (week 5 found the same leaf size). AdaBoost picks **weak** learners, trees of depth 4 with a small `learning_rate` and many rounds, because boosting reduces bias step by step and a strong base tree leaves it only noise to fit. Tuning helped most where the defaults overfit: the fully grown forests went from gaps of +0.09 and +0.11 to about +0.01. AdaBoost with stumps was already well behaved. I would choose AdaBoost and report **0.684 ± 0.014** (nested CV, accuracy), with the caveat that the random forest (0.681) and the oblique forest (0.680) are statistically indistinguishable from it. *(The week 5 forest scored 0.678 with a different search space; this week `n_estimators` is fixed at 200 and `max_depth` is an integer from 2 to 30.)* The oblique forest gains nothing over the ordinary one (+0.001 the other way, p = 0.92). COMPAS's signal is mostly "more priors, younger", which one-feature splits already express well, so slanted splits have little to add.

**Challenge 1 -- feature engineering.** Besides the course's `priors_per_adult_year` (priors / years since 18), I added three features. `has_juvenile_record` is 1 if any juvenile count is above 0 (70% of those people reoffend, against 41% of the rest). `young_x_priors` is `priors_count` for people under 25 and 0 otherwise, an interaction. `priors_bin` puts priors into the fixed bins 0 | 1 | 2–3 | 4–9 | 10+. **None of them needs anything learned from the data**: they are fixed formulas with fixed constants, so they can be computed before the split and on new data. A missing input gives a missing feature, which the pipeline's imputer then fills inside each fold. A bin edge such as "the median of priors" would be learned, and would have to become a `Pipeline` step. (To be transparent: I chose the round bin edges after looking at reoffending rates by priors in the development set. That is a human choice informed by the data, so it is not perfectly blind, but the test set was never involved.)

| Model, nested CV (accuracy) | Without | With the 4 features | Difference (folds won, corrected p) | ROC-AUC without → with |
|---|---|---|---|---|
| Logistic regression | 0.671 ± 0.013 | 0.677 ± 0.014 | +0.006 (3/5, p = 0.47) | 0.717 → 0.723 |
| Random forest | 0.681 ± 0.015 | 0.673 ± 0.006 | −0.008 (1/5, p = 0.33) | 0.727 → 0.726 |
| AdaBoost | 0.684 ± 0.014 | 0.676 ± 0.011 | −0.007 (1/5, p = 0.25) | 0.728 → 0.725 |

Only the **linear model** gains, and by less than the fold std. That makes sense: logistic regression cannot divide, multiply or bin features on its own, so handing it a rate, an interaction and a step function gives it shapes it could not build. With the extra, correlated columns its tuner also chose stronger regularization (`C` 4.5 → 0.39). The tree ensembles already approximate these combinations with splits on `age` and `priors_count`, and they lose a little: four near-duplicates of `priors_count` crowd the random feature subsets at each split. The features stay **off** in `config.yaml` (`engineered_features: []`). They can be switched on per run.

**Challenge 2 -- boosting a model that isn't a tree.** `AdaBoostClassifier(estimator=LogisticRegression(max_iter=2000))` is **worse** than one logistic regression on the same folds: accuracy 0.652 ± 0.016 against 0.672, and ROC-AUC 0.702 against 0.717. With 200 rounds and `learning_rate=0.1` it scores 0.658 and 0.702. With the default 50 rounds it kept only 21: scikit-learn stops boosting once a weighted learner is no better than chance. And the result is **not linear**. A weighted sum of linear *scores* would still be linear, so boosting could add nothing. But AdaBoost (SAMME) sums each learner's *vote* (±1), not its score, so the ensemble is a weighted vote of half-planes, a piecewise-constant function. Its out-of-fold probabilities take only 158 distinct values, against 3,755 for the single logistic regression. That staircase throws away the fine ranking a logistic regression gives (hence the lower ROC-AUC), and the re-weighting keeps pushing new learners towards noisy rows that a linear boundary cannot separate anyway. Boosting pays off with weak, flexible learners (shallow trees), not with a stable linear model.

### Week 7 -- Scoreboard (ROC-AUC)

**Selection metric: ROC-AUC.** This model produces a **risk score**, like COMPAS's own decile score, and what matters first is whether it **ranks** people who reoffend above people who don't. ROC-AUC measures exactly that ranking over every threshold, so it does not prejudge the separate decision of where to cut. Each cut-off trades false accusations against missed reoffenders, and the fairness check shows that the trade-off falls unevenly on different groups. Accuracy only counts which side of 0.5 each person lands on: it is noisier from fold to fold, and it rewards a model for matching the 0.5 cut, not for ranking well.

Every model, tuned where it has a search space (30 Optuna trials, 5 inner folds), on the same 5 outer folds. **Untuned** = the same model with its `config.yaml` params, **floor** = the dummy (0.500 on every fold). Accuracy and recall come from the nested out-of-fold probabilities at threshold 0.5.

| Model | Untuned: CV ROC-AUC | Tuned: nested CV ROC-AUC (gap) | Best trial | Accuracy | Recall | Chosen hyperparameters |
|---|---|---|---|---|---|---|
| Dummy (floor) | 0.500 ± 0.000 | — | — | — | — | — |
| Naive Bayes | 0.685 ± 0.011 | 0.696 ± 0.011 (+0.001) | 0.696 | 0.606 | 0.202 | `var_smoothing=0.49` |
| Logistic regression | 0.717 ± 0.014 | 0.718 ± 0.014 (+0.002) | 0.718 | 0.673 | 0.521 | `C=28.8` |
| Decision tree | 0.614 ± 0.018 | 0.720 ± 0.015 (+0.013) | 0.719 | 0.675 | 0.575 | `max_depth=11`, `min_samples_leaf=120`, `gini` |
| Random forest | 0.684 ± 0.019 | 0.725 ± 0.016 (+0.020) | 0.728 | 0.679 | 0.558 | `max_depth=15`, `min_samples_leaf=36`, `max_features=0.5` |
| Oblique random forest | 0.688 ± 0.018 | 0.727 ± 0.015 (+0.018) | 0.728 | 0.679 | 0.553 | tree `max_depth=28`, `min_samples_leaf=61`, `n_components=8` |
| AdaBoost | 0.723 ± 0.016 | 0.727 ± 0.014 (+0.014) | 0.729 | 0.683 | 0.561 | base `max_depth=4`, `n_estimators=55`, `learning_rate=0.17` |
| **Gradient boosting** | 0.728 ± 0.015 | **0.728 ± 0.016** (+0.012) | 0.730 | 0.683 | 0.566 | `n_estimators=190`, `learning_rate=0.016`, `max_depth=4`, `min_samples_leaf=76`, `subsample=0.60` |
| **HistGradientBoosting** | 0.719 ± 0.019 | **0.728 ± 0.016** (+0.011) | 0.730 | 0.685 | 0.568 | `max_iter=452`, `learning_rate=0.020`, `max_leaf_nodes=4`, `min_samples_leaf=24`, `l2_regularization=0.0007` |
| Gradient boosting + feature selection | 0.723 ± 0.011 | 0.726 ± 0.013 (+0.005) | 0.726 | 0.679 | 0.557 | `max_depth=2`, 394 trees, `learning_rate=0.013`, `min_votes=1` |
| Logistic regression + feature selection | 0.716 ± 0.011 | 0.716 ± 0.012 (+0.001) | 0.716 | 0.674 | 0.503 | `C=65`, `min_votes=1` |

**What the corrected paired t-test can separate** (best mean, HistGradientBoosting, against each model, same 5 folds). The **top four are tied**: gradient boosting (+0.0001, p = 0.88), AdaBoost (+0.0007, p = 0.74) and the oblique forest (+0.0005, p = 0.85). But the boosters are **really better** than the random forest (+0.003, 5 of 5 folds, p = 0.006), the tuned tree (+0.008, p = 0.03), logistic regression (+0.010, p = 0.047) and Naive Bayes (+0.032, p = 0.006). The random forest result shows why ROC-AUC is the better selection metric: a gap of only 0.003 is detectable because it appears in every fold, while in week 6 a lead of the same size on accuracy (AdaBoost over the random forest, +0.003, p = 0.47) could not be told apart from noise. Week 6's "everything ties at 0.68" was partly the noise of accuracy.

**Reading the scoreboard.** Every model is far above the floor, but the ceiling is low: the best ROC-AUC is 0.728, against 0.718 for a plain logistic regression. The remaining signal in these 10 features is small, so more model gains about one point. **Tuning mattered for the models that overfit by default** (tree +0.106, forests +0.04), and hardly at all for the boosters and logistic regression: gradient boosting's defaults (100 depth-3 trees, `learning_rate=0.1`) are already a good setting here. Both boosters chose **small trees and small steps** (depth 4 / 4 leaves, learning rates around 0.02, hundreds of trees), which is the week 7 recipe against boosting's overfitting. Naive Bayes ranks reasonably (0.696), but at 0.5 it flags almost nobody (recall 0.20). A likely reason: it assumes features are independent given the class, so correlated features such as `age` and `age_cat` are counted twice, which pushes its probabilities to the extremes and moves the 0.5 cut-off. The same trait gives it a misleadingly low false positive rate (0.09 / 0.04 for African-American / Caucasian defendants), because it simply flags few people.

**Fairness (nested out-of-fold, threshold 0.5).** For the tied top models, the false positive rate is 0.29–0.30 for African-American defendants and 0.15–0.16 for Caucasian defendants, the same gap of about 0.14 as every week since week 3. COMPAS's own score gives 0.45 / 0.23 on the same rows. Better ranking did not narrow the gap. Closing it would need a decision about thresholds, not a better model.

**Part B1 -- feature selection.** `VoteSelector` was tuned with the model (`select__min_votes` from 1 to 3) and compared on nested scores with `corrected_paired_test()`. It helped **neither** model: gradient boosting 0.726 vs 0.728 (−0.002, 1 of 5 folds, p = 0.34), logistic regression 0.716 vs 0.718 (−0.002, p = 0.57). The tuner chose `min_votes=1` on the whole development set (keep every feature with at least one vote), and the outer folds went both ways (1, 2 or 3), with no consistent effect on the score. With only 10 features after preprocessing there is little to remove, and the leak-free version is expensive, because the three voters are refit in every fold of every trial (about 5–9 minutes per run here). It stayed **off** that week. On a dataset with hundreds of features the same step would matter much more. *(Week 8 revisits this: caching makes it cheap, and the smaller model turns out to score the same.)*

**Best model at the end of week 7:** tuned gradient boosting, **0.728 ± 0.016** ROC-AUC (nested CV). It is tied with HistGradientBoosting (0.728, faster), AdaBoost (0.727) and the oblique forest (0.727), and it is significantly better than the random forest, the tree, logistic regression and Naive Bayes. This is the number I would report, with the tie stated. If speed mattered, I would pick HistGradientBoosting at no cost in score.

### Week 8 -- Feature selection

Tuned XGBoost (30 Optuna trials, 5 inner folds), ROC-AUC, nested CV, the same 5 outer folds as every week. Each selection run is compared with the run without selection, using `corrected_paired_test()`.

| Run | Untuned: CV ROC-AUC | Tuned: nested CV ROC-AUC (gap) | Difference vs no selection (folds won, corrected p) | Features kept by the final model | Run time |
|---|---|---|---|---|---|
| No selection | 0.725 ± 0.018 | 0.727 ± 0.015 (+0.010) | — | all 10 | 72 s |
| **`vote`, `min_votes` tuned** | 0.721 ± 0.013 | **0.726 ± 0.013 (+0.005)** | −0.001 (3/5, p = 0.57) | `age`, `priors_count` | 111 s |
| `vote`, `min_votes=2` fixed | 0.721 ± 0.013 | 0.726 ± 0.012 (+0.004) | −0.001 (3/5, p = 0.67) | `age`, `priors_count` | 89 s |
| `kbest`, `k` tuned | 0.721 ± 0.015 | 0.728 ± 0.014 (+0.009) | +0.001 (3/5, p = 0.71) | 9: all except `c_charge_degree_was_missing` | 72 s |
| `from_model`, threshold tuned | 0.721 ± 0.012 | 0.728 ± 0.014 (+0.009) | +0.001 (4/5, p = 0.42) | 5: `age`, `priors_count`, `sex`, `age_cat`, `c_charge_degree` | 169 s |

**Is the smaller model as good? As far as these folds can tell, yes.** With `min_votes` fixed at 2, every outer fold keeps exactly `age` and `priors_count`, and those 2 columns score 0.726 against 0.727 for all 10. The difference (−0.001, p = 0.67) is far smaller than the fold-to-fold std. A non-significant test on 5 folds does not prove the two are equal, but it does say that any cost is too small to detect here. The chosen XGBoost also becomes very simple: on the 2 features the tuner picks depth-1 trees (`max_depth=1`), so each tree uses one feature and the model has no interactions between them. The fixed run is the honest number for that small model, because in the tuned run the outer folds chose different `min_votes`. **I keep selection on**, with `vote`: same score, 2 inputs instead of 10, and a model that is much easier to explain. It also stops using `sex` directly. That does not make it fair, though: the false positive rates stay about the same, 0.30 / 0.17 for African-American / Caucasian defendants against 0.31 / 0.18 without selection, and those gaps are similar. The cost is small, about 40 extra seconds per run.

**XGBoost against the other models.** Without selection it ties with HistGradientBoosting (−0.001, p = 0.62), gradient boosting (−0.001, p = 0.62) and AdaBoost (−0.000, p = 0.91), and beats logistic regression (+0.010, 5 of 5 folds, p = 0.02). Against the random forest it wins all 5 folds, but this time the test cannot call it (+0.002, p = 0.29). Its aggressive tree settings (`max_depth=7`) are held back by a large `gamma` (3.4) and `min_child_weight` (10.8). Another strong model on the same 10 features hits the same ceiling of about 0.73: the limit is the information in the features, not the booster.

**B1 -- other selectors.** All tie, on the same folds, with each other and with no selection. **`kbest` is the fastest for the same score** (72 s, no slower than no selection), but it removes almost nothing (it keeps 9 of 10). `from_model` is the slowest here (169 s) and keeps 5. `vote` is the only one that gets down to 2, at 111 s. So the choice is not about score: I prefer `vote` because it gives the smallest model.

**B2 -- how many votes?** Nested CV chose `min_votes` 2, 1, 3, 1, 1 in the five outer folds, so **not** the same in every fold. That does not mean the selection is unstable, because the choice barely changes anything. On each fold's training rows, one vote already gives `age` and `priors_count` (plus `age_cat` on one fold), and two votes give exactly those two features every time. The tuner is choosing between almost identical models, so the differences come from noise.

**What I found in the voters.** Three votes are almost never reached in this pipeline, because the L1 logistic regression hardly votes. With `C=0.01` on our min-max-scaled columns, it shrinks every coefficient to zero on 4 of the 5 folds. With the course's robust scaler, the same L1 model keeps exactly `age` and `priors_count` on all 5 folds. So `min_votes=3` only works here through the guard I added (keep the most-voted features), and it is really a two-voter system. The kept features would be the same under either scaler, so I left the scaler alone. Still, the fixed `C` makes this voter depend on the scale of the data, and that is worth remembering if the scaler changes.

**Current best model:** tuned XGBoost with `vote` selection, the model in `config.yaml`: **0.726 ± 0.013** ROC-AUC (nested CV) on `age` and `priors_count`. It is statistically tied with the larger boosters (0.728 at best), and it is the one I would use because it is the simplest. If only the highest mean mattered, I would report `from_model` or HistGradientBoosting at 0.728, but those leads are noise.

### Week 10 -- Missing values

**Where the gaps are (COMPAS development set).** About 1 person in 7 (777 of 5,771, 13%) has at least one gap, mostly in `priors_count` (397), `c_charge_degree` (192) and `age` (120). Deleting those rows is not an option for a deployed model: a new person with a gap still needs a prediction. `priors_count` is the most often missing column and also the strongest predictor, so dropping it is not an option either. For `priors_count` and `c_charge_degree` the gap itself carries information, so they also keep their `_was_missing` flag. **No imputer uses the label**: filling a gap with the mean of the people with the same outcome would leak in cross-validation (the filled value *is* the label), and it cannot run at all on a new person, whose label is unknown.

**A1 -- the imputers, on the same folds** (tuning off, ROC-AUC, corrected paired test against `median`):

| Imputer | Configured model (XGBoost + `vote`) | vs `median` | Logistic regression (all features) | vs `median` | Run time (XGBoost) |
|---|---|---|---|---|---|
| **median** (kept) | **0.721 ± 0.013** | — | **0.717 ± 0.014** | — | fast |
| mean | 0.720 ± 0.013 | −0.001 (1/5, p = 0.82) | 0.717 ± 0.013 | −0.000 (1/5, p = 0.53) | fast |
| KNN (k = 5) | 0.720 ± 0.012 | −0.001 (2/5, p = 0.79) | 0.716 ± 0.013 | −0.002 (0/5, p = 0.17) | fast |
| iterative (MICE) | 0.721 ± 0.014 | +0.000 (2/5, p = 0.85) | 0.718 ± 0.013 | +0.001 (5/5, p = 0.21) | fast |
| random_forest (missForest) | 0.722 ± 0.013 | +0.001 (2/5, p = 0.65) | 0.718 ± 0.014 | +0.000 (3/5, p = 0.79) | ~45 s, the slowest by far |

**I keep `median`.** No imputer is distinguishable from it: every difference is a thousandth or less, and every p-value is above 0.15. `median` is also the fastest, the simplest to explain, and the cheapest thing to carry inside a deployed model (one number per column). The reason is the one the course shows: a model-based imputer is only as good as the columns it can see, and our numeric imputer only sees five numeric columns that say little about each other. With `iterative`, `mean` and `random_forest` the vote selection keeps `age_cat` as a third feature, but the score doesn't move.

**B2 -- letting the imputer see the categorical columns.** I hid 200 known values of `age` and of `priors_count` in fold 1's validation rows, fitted each imputer on the training rows, and measured the mean absolute error of its guesses. I did this once with the numeric columns only, and once with the categorical columns added as one-hot context (`ContextImputer` in `wk10/run_imputers.py`, which returns only the numeric columns).

| Error of the guesses (MAE) | median | KNN | iterative | random_forest |
|---|---|---|---|---|
| `age`, numeric columns only | 9.2 years | 10.8 | 9.0 | 9.0 |
| `age`, + categorical columns | 9.2 | **4.4** | **3.9** | **3.9** |
| `priors_count`, numeric only | 3.1 priors | 3.2 | 3.3 | 3.1 |
| `priors_count`, + categorical | 3.1 | 3.1 | 3.1 | 3.0 |

Seeing `age_cat` **more than halves** the error on `age`, which is exactly the course's point. `priors_count` stays hard: nothing we have predicts it. The model barely notices, though: ROC-AUC 0.724 (KNN) and 0.724 (iterative) with the context, against 0.720 and 0.721 without it (+0.004, p = 0.20 and +0.002, p = 0.14). That is consistent in direction, but not a detectable gain. Better guesses for `age` don't help much because the model already has `age_cat`, and the gaps that matter, in `priors_count`, can't be guessed well by anything.

**B1 -- multiple imputation (MICE in full).** I trained m = 5 pipelines with `IterativeImputer(sample_posterior=True, random_state=i)` in each fold and averaged their probabilities. The 5 completed datasets disagree about a missing `priors_count` by about ±3.8 priors on average: that spread is the uncertainty a single imputation hides. The average reaches 0.722 ± 0.013, against 0.721 for one iterative dataset and 0.721 for `median` (+0.001, p = 0.54). **It is not worth m times the cost here** (63 s against 10 s): the ensemble's gain is far below the fold-to-fold noise.

**B3 -- random-forest proximities, from scratch.** I grew a forest (300 trees) on fold 1's training rows, using their labels. Each validation row with a hidden value is first filled with the training median and passed through the trees. Its proximity to each training row is the share of trees where the two land in the same leaf, and the gap becomes the proximity-weighted average of the training rows' values, repeated 3 times. For `age` the error falls from 6.4 to 4.3 years over the iterations (better than any numeric-only imputer, close to the imputers that see the categories). For `priors_count` it gets slightly worse (3.08 → 3.18), so there is no gain there either. **Could it run in `predict.py`?** Only partly. The forest uses only the *training* rows' labels, so a new person's label is never needed. But the model would have to carry the whole training set (its values and its leaves) to compute proximities, and every prediction would need several passes through the forest. The theory's "one copy per class" trick is needed when the forest is grown with the row's own label, which is how the method works on the training data, and that label is unknown for a new person. `IterativeImputer` with a random forest gets the same idea (forests learn the relationships) into a fitted, self-contained step.

### Predicting new data

```bash
python main.py                                                     # train, evaluate, save models/model.joblib
python predict.py data/compas_new_people.csv predictions.csv       # predict 20 new people
python predict.py data/adult_new.csv out.csv --config config_adult.yaml   # same for Adult (models/adult_model.joblib)
```

The output CSV has **one row per input row**, in the same order: the `id`, the `probability` of the positive class, and the 0/1 `prediction` at 0.5. For `compas_new_people.csv` that is 20 rows in and 20 out (6 predicted to reoffend). The `"?"` in `priors_count` (row 3) and the missing age (row 7) are filled by the saved imputer. The never-seen charge degree `"X"` (row 11) is accepted by the encoder. The same person twice (rows 14–15) gets the same prediction, and both rows are kept. No training happens: the saved pipeline holds every learned value (medians, encodings, selected features, trees). For Adult, `predict.py` reproduces `predictions/adult_new_predictions.csv` exactly. A `.joblib` file can run code when loaded, so only load models you trust, and load them with the library versions they were saved with (`predict.py` warns when these differ).

## Beyond COMPAS (week 9, option A: Adult income)

**The question:** does a person earn more than 50,000 USD a year? The data is the 1994 US Census (UCI *Adult*), subsampled and extended for the course. `data/adult_train.csv` has 6,000 labelled people (24% earn more than 50K). `data/adult_new.csv` has 2,000 people without a label. Column descriptions are in `data/ADULT_README.md`.

**How to run it:**

```bash
python main.py                      # COMPAS, exactly as before (config.yaml)
python main.py config_adult.yaml    # Adult income
```

The Adult run writes its reports and scoreboard to `results/adult/`, so they never mix with COMPAS's. It also writes the predictions for the new people to **`predictions/adult_new_predictions.csv`** (`id`, `probability`, `prediction`; 2,000 rows in, 2,000 out). `wk09/run_adult.py` reruns every comparison below.

**Decisions in `config_adult.yaml`.** I audit `sex` (the `data.sensitive_attr`). I keep `race`, the second sensitive attribute, out of the features too, so that no prediction depends on it directly, although `relationship` ("Husband" / "Wife") still carries sex indirectly. `fnlwgt` is dropped: it is a census sampling weight, not a trait of the person. `education` is dropped as redundant with `education_num`, which keeps the order. The `?` placeholders become missing values. The unknown `workclass` and `occupation` are the same people, and they earn more than 50K half as often, so one `workclass_was_missing` flag records that. `native_country` and `v01`–`v06` are imputed without a flag. The validity rules are Adult's own (age 17–90). `capital_gain`'s 99999 is the census top code, a real value, so it is kept. Target encoding handles `native_country`'s 41 mostly rare values. The 25 undocumented columns `v01`–`v25` are all listed as features, and feature selection decides about them.

### What broke, why, and how I fixed it

Each fix is general and driven by the config, so `src/` names no Adult column. After all the fixes, the COMPAS report (`python main.py`, tuning off) is **identical, character for character**, to the one from before today.

| # | What broke | Why | Fix |
|---|---|---|---|
| 1 | `main.py` could only run COMPAS | `load_config()` always read `config.yaml` | the config is an optional argument: `python main.py config_adult.yaml` (default `config.yaml`) |
| 2 | Adult's `?` stayed as a category of its own (379 in `workclass`, 381 in `occupation`, 86 in `native_country`). **No error**, just a wrong category | placeholders were only turned into missing values in the columns that had a `canonical_maps` entry, and on COMPAS every placeholder happened to be in such a column | new `placeholders_to_missing()`: placeholders become missing in **every** text column. Checked: 379 / 381 / 86 missing, no `?` left, in both files |
| 3 | COMPAS's validity rule would have erased 74 valid people | `age: {min: 18}` is COMPAS's rule (adult defendants); Adult has 17-year-olds | each dataset's rules live in its own config (`age: 17–90`, hours 1–99, gains ≥ 0) |
| 4 | `clean_dataset()` crashed on a config without every COMPAS key | it read `cleaning["numeric_as_text"]` etc. with `[]` | every cleaning setting is optional (empty by default) |
| 5 | COMPAS's own repairs ran on every dataset | `recover_juv_fel_count()` and `fix_age_cat()` were always called and only skipped because Adult lacks those columns | they run only when named in `cleaning.derived_fixes` (the COMPAS config lists both) |
| 6 | The fairness report crashed (`KeyError: 'score_text'`) | it always compared with COMPAS's score, and its wording ("did NOT reoffend") was COMPAS's | a `fairness` section in the config: an optional `reference` score and the wording. Adult has no reference, so only the model is audited |
| 7 | `score_text` was always kept for the audit | `split_features_target()` named it in the code | `data.audit_columns` in the config |
| 8 | The final model could not predict new people | the pipeline only evaluated and refit | new `src/predict.py`: same cleaning, engineered features and feature split, **no** `drop_duplicate_rows()`. It writes `id`, probability and 0/1 prediction, and fails loudly if rows out ≠ rows in |
| 9 | The scoreboard would have mixed both datasets | one `results/` folder | each config has its own `output` folder (`results/adult/`) |

I found most of these by reading `src/` for COMPAS column names and hard-coded steps, then confirmed the silent ones on Adult's data before fixing them: #2 (the 379 `?` were still there) and #3 (74 ages became missing). I also checked the course's warning that an unlisted column is silently left out: every Adult column reaches the model, and the "features kept" line ends every run.

### Results (Adult, development set: 4,800 people; test set locked)

ROC-AUC is the selection metric. Every tuned run uses 30 Optuna trials and nested CV, on the same 5 outer folds. Accuracy and F1 (class 1, threshold 0.5) are the nested out-of-fold means. Targets: ROC-AUC ≥ 0.90, accuracy ≥ 0.855, F1 ≥ 0.66, all in the same run.

| Run | ROC-AUC | Accuracy | F1 | Features kept | Targets |
|---|---|---|---|---|---|
| No model (dummy) | 0.500 | 0.761 | 0.000 | — | — |
| Decision tree, default | 0.726 ± 0.015 | 0.801 | 0.583 | 36 | no |
| Logistic regression, tuned | 0.893 ± 0.013 | 0.844 | 0.637 | 36 | no |
| HistGradientBoosting, tuned | 0.906 ± 0.013 | 0.861 | 0.666 | 36 | yes |
| XGBoost, tuned, no selection | 0.907 ± 0.016 | 0.862 | 0.669 | 36 | yes |
| **XGBoost, tuned + `vote` selection** (`config_adult.yaml`) | **0.910 ± 0.014** | **0.857** | **0.668** | **8** | **yes** |
| XGBoost, tuned + `kbest` | 0.909 ± 0.015 | 0.859 | 0.671 | 9 | yes |
| XGBoost, tuned + `from_model` | 0.911 ± 0.016 | 0.863 | 0.677 | 9 | yes |

**Corrected paired tests (ROC-AUC, same folds).** Selection helps here, unlike on COMPAS. `vote` beats no selection by +0.003 in **5 of 5 folds** (p = 0.06), `kbest` by +0.003 in 5 of 5 (p = 0.03), and `from_model` by +0.004 in 5 of 5 (p = 0.10). The three selectors tie with each other (`from_model` vs `vote`: +0.001, p = 0.68). XGBoost and HistGradientBoosting tie (+0.001, p = 0.80). Both clearly beat logistic regression (XGBoost +0.013, 5 of 5 folds, p = 0.002).

**Features kept by the final model (`vote`):** `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`, `marital_status`, `occupation`, `relationship`. **None of the 25 `v` columns survives** in any of the three selectors' final models (`kbest` and `from_model` keep these 8 plus `workclass`). The undocumented columns carried no signal the voters could find. They were 25 of 36 inputs, so dropping them is also what makes selection pay off here: on COMPAS there was no noise to remove. `native_country` is dropped as well. The tuner chose `min_votes=1` in all five outer folds, so on this data keeping everything with at least one vote is enough to get rid of the noise. (The L1 voter still hardly votes with our min-max scaler, as in week 8.)

**Fairness (`sex`, nested out-of-fold, threshold 0.5).** Of the people who earn 50K or less, the final model wrongly predicts more for 2.4% of women and 8.8% of men. The model flags men far more often, which mirrors the data, where men earn more than 50K much more often. A gap in false positive rates is still a gap, and with `relationship` and `marital_status` in the model, sex enters indirectly even though it is not a feature.

**What was hardest, and what got the targets.** The hardest assumption to find was the placeholder rule (#2). It raised no error: the pipeline ran and scored well with a fake `"?"` category, because on COMPAS every placeholder had sat in a mapped column by coincidence. Only looking at `value_counts()` after cleaning exposed it. The silent ones are the dangerous ones: wrong categories, an unlisted column, a validity rule that erases valid rows. The targets came from three things together. First, correct cleaning: real missing values, a flag for the informative ones, valid ages kept. Second, a tuned booster: the default tree reaches only 0.726 ROC-AUC, and logistic regression falls short at 0.893 / 0.844 / 0.637. Third, feature selection removing the 25 noise columns, which added a small but consistent gain. The final run passes all three targets, with accuracy closest to its line (0.857 ± 0.014 against 0.855). `from_model` gives more margin (0.863), but it ties with `vote` on the selection metric, so I kept the same selector as COMPAS. The final model is refit on the 4,800 development rows only: the locked test set stays untouched, as it is for COMPAS.
