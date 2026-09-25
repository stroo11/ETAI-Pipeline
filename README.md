# Predictive Pipeline -- ETAI -- Name: Jovaldo Rodrigues 20260573

This project predicts whether a defendant reoffends within two years (`two_year_recid`) using ProPublica's COMPAS dataset. Race is never used as a model input. It is kept aside only to check whether the model makes more mistakes for some groups than for others, and to compare our model with COMPAS's own risk score.

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

**Current best model:** the decision tree with a maximum depth of 5, with 0.675 average test accuracy and a small gap between train and test. It beat logistic regression on 10 of the 15 splits, but only by 0.003 on average, so in practice the two models are tied. Logistic regression remains a good alternative that is also easier to interpret.
