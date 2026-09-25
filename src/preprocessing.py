"""
Preprocessing -- week 3: from diagnosis to a deployable recipe.

Raw data in, model-ready split out, in two stages:

1. `clean_dataset` -- deterministic cleanup driven by the EDA diagnosis in config.yaml (canonical categories, placeholder / domain-rule violations -> NaN, de-duplication, redundant columns dropped). Nothing here is learned from the data, so it is safe to run on the whole dataset before the split.
2. `build_preprocessor` -- everything that IS learned (imputed medians/modes, encoder categories, scaler mean/std) lives in a ColumnTransformer that is fit on the training rows only, then applied unchanged to test / inference rows.

None of these functions require the target column, so the same code runs on unlabeled data at inference time.

`sensitive_attr` (race) is still kept out of the model's input features entirely -- it is returned in `extras` only to audit fairness afterwards. See src/evaluate.py:fairness_report.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


# ---------------------------------------------------------------- 1. cleaning

def canonicalize_categories(df: pd.DataFrame, canonical_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    """Strip whitespace, map known spelling/casing variants to one label, and turn placeholder tokens into NaN.
    A value that is neither a known variant nor a placeholder is kept as-is, so a genuinely new category doesn't silently disappear."""
    out = df.copy()
    for col, mapping in canonical_maps.items():
        if col not in out.columns:
            continue
        missing = out[col].isna()
        cleaned = out[col].astype(str).str.strip()
        out[col] = cleaned.str.lower().map(mapping).fillna(cleaned)
        out.loc[missing | out[col].isin(placeholder_tokens), col] = np.nan
    return out


def apply_domain_rules(df: pd.DataFrame, domain_rules: dict) -> pd.DataFrame:
    """Values outside the valid [min, max] range of a column become NaN, so they get imputed like any other missing value."""
    out = df.copy()
    for col, bounds in domain_rules.items():
        if col not in out.columns:
            continue
        low, high = bounds.get("min", -np.inf), bounds.get("max", np.inf)
        out.loc[out[col].notna() & ~out[col].between(low, high), col] = np.nan
    return out


def recover_juv_fel_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extra finding (not in the EDA notebook): juvenile_total is never missing and always equals juv_fel + juv_misd + juv_other, so a missing juv_fel_count can be recovered exactly instead of imputed.
    Must run BEFORE juvenile_total is dropped as redundant, and AFTER domain rules (juvenile_total is itself -1 in the rows where juv_fel_count is -1, so those stay NaN).
    """
    out = df.copy()
    needed = {"juv_fel_count", "juvenile_total", "juv_misd_count", "juv_other_count"}
    if not needed.issubset(out.columns):
        return out
    recovered = out["juvenile_total"] - out["juv_misd_count"] - out["juv_other_count"]
    fillable = out["juv_fel_count"].isna() & (recovered >= 0)
    out.loc[fillable, "juv_fel_count"] = recovered[fillable]
    return out


def fix_age_cat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extra finding (not in the EDA notebook): a few rows have an age_cat that contradicts age (e.g. age 51 labelled "Less than 25").
    age_cat is a deterministic function of age -- the dataset's own convention is <25, 25-44, >=45 -- so wherever age is valid it is recomputed from age. Where age is missing, the original age_cat is kept.
    """
    out = df.copy()
    if not {"age", "age_cat"}.issubset(out.columns):
        return out
    age = out["age"]
    derived = pd.Series(
        np.select([age < 25, age < 45], ["Less than 25", "25 - 45"], "Greater than 45"),
        index=out.index,
    )
    out.loc[age.notna(), "age_cat"] = derived[age.notna()]
    return out


def clean_dataset(df: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
    """
    Apply the EDA diagnosis (config.yaml -> preprocessing.cleaning). Target-agnostic and fit-free, so it is safe on the full dataset and on label-free inference data.
    """
    out = df.copy()
    placeholder_tokens = set(cleaning["placeholder_tokens"])

    # numeric columns that loaded as text because of placeholder tokens like "-"
    for col in cleaning["numeric_as_text"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    out = apply_domain_rules(out, cleaning["domain_rules"])
    out = canonicalize_categories(out, cleaning["canonical_maps"], placeholder_tokens)
    out = recover_juv_fel_count(out)
    out = fix_age_cat(out)

    # exact duplicates and repeated ids point at the same 72 rows -- keep the first occurrence
    out = out.drop_duplicates()
    if "id" in out.columns:
        out = out.drop_duplicates(subset="id", keep="first")

    # redundant columns (multicollinearity), dropped only after they've been used above
    out = out.drop(columns=[c for c in cleaning["redundant_columns"] if c in out.columns])

    return out


# ------------------------------------------------------- 2. features / target

def mnar_columns(imputation: dict) -> list:
    """Columns whose missingness is informative (MNAR) -- they get a `<col>_was_missing` flag."""
    return [col for col, plan in imputation.items() if plan.get("indicator")]


def add_missingness_indicators(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each MNAR column, BEFORE that column gets imputed."""
    out = df.copy()
    for col in columns:
        out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, target: str, sensitive_attr: str, drop_columns: list, imputation: dict):
    """
    Returns (X, y, extras). On label-free inference data `y` is None -- nothing downstream requires the target.
    `extras` holds the sensitive attribute and COMPAS's own score, kept for the fairness audit, never used as model inputs.
    """
    df = add_missingness_indicators(df, mnar_columns(imputation))
    y = df[target] if target in df.columns else None
    extras = df[[c for c in [sensitive_attr, "score_text"] if c in df.columns]].copy()
    excluded = set([target, sensitive_attr] + drop_columns)
    X = df[[c for c in df.columns if c not in excluded]]
    return X, y, extras


def split_train_test(X, y, extras, test_size: float, random_state: int):
    """Stratified train/test split, keeping X, y and extras row-aligned -- week 2's original job."""
    return train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)


# ------------------------------------------------------ 3. leak-safe transformer

_SCALERS = {
    "none": lambda: "passthrough",
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "target": lambda: TargetEncoder(random_state=0),  # unseen categories fall back to the target mean
}


def build_preprocessor(prep_config: dict) -> ColumnTransformer:
    """
    Leak-safe ColumnTransformer: every imputer / encoder / scaler inside it is fit on the training rows only when the surrounding Pipeline is fit.
    Every encoder tolerates categories it has never seen, so a new value at inference time doesn't crash the pipeline.
    """
    encoder_name, scaler_name = prep_config["encoder"], prep_config["scaler"]
    if encoder_name not in _ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder_name}. Options: {list(_ENCODERS)}")
    if scaler_name not in _SCALERS:
        raise ValueError(f"Unknown scaler: {scaler_name}. Options: {list(_SCALERS)}")

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", _SCALERS[scaler_name]()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", _ENCODERS[encoder_name]()),
    ])
    indicator_columns = [f"{c}_was_missing" for c in mnar_columns(prep_config["imputation"])]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, prep_config["numeric_features"]),
        ("categorical", categorical_pipeline, prep_config["categorical_features"]),
        ("indicators", "passthrough", indicator_columns),
    ])
