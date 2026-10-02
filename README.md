# Predictive Pipeline -- ETAI -- Name: Jovaldo Rodrigues 20260573

This project predicts whether a defendant reoffends within two years (`two_year_recid`) using ProPublica's COMPAS dataset. Race is never used as a model input. It is kept aside only to check whether the model makes more mistakes for some groups than for others, and to compare our model with COMPAS's own risk score.

Run it with `python main.py`. Everything tunable lives in `config.yaml`.

## Pipeline progress

| Week | What changed | How it is evaluated | Best model (accuracy) |
|---|---|---|---|
| 2 | Baseline: drop missing rows, one-hot everything | one 80/20 split | Logistic regression, 0.680 (test) |
| 3 | EDA-driven cleaning, imputation by missingness mechanism, leak-safe `Pipeline` | one 80/20 split (+ 15 repeated splits) | Decision tree, depth 5, 0.665 (test) |
| 4 | Row-preserving cleaning, locked test set, cross-validation, dummy + random forest, configurable imputer (KNN) and log1p | stratified 5-fold CV on the development set | Random forest, `min_samples_leaf=40`, **0.683 ± 0.016** (CV) |

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

**What did help: regularizing the random forest.** Requiring at least 40 people in every leaf stops each tree from memorizing individuals. This raised CV accuracy from 0.649 to 0.683 and closed the train–validation gap from +0.083 to +0.005. Because I picked this setting from 17 candidates, I re-checked it on 25 *new* folds that played no part in the choice. It still beat the week 3 tree by +0.76 percentage points, winning 21 of 24 folds (corrected p = 0.08). That is a small but consistent improvement, and it is the new model in `config.yaml`.

**Fairness (out-of-fold, development set):** our models still make far fewer false accusations than COMPAS, which wrongly flags 45% of African-American and 23% of Caucasian defendants who did not reoffend, against 26% / 13% for logistic regression and 30% / 16% for the tuned forest. The gap between the two groups (about 0.13) is the same in every model and has not shrunk since week 3. The tuned forest predicts "reoffends" more often, so it catches more people who do reoffend (recall 0.569 vs 0.511), but its false positive rate is 1 to 4 points higher in every group. This is the cost of its higher accuracy, and the next step is to choose the decision threshold based on what each type of error costs.

**Current best model:** a random forest with 300 trees and `min_samples_leaf=40`: **0.683 ± 0.016** CV accuracy, train–validation gap +0.005, and roughly 1 point better than the week 3 tree, confirmed on fresh folds. The depth-5 tree (0.674) and logistic regression (0.672) remain close, and they are simpler to explain.
