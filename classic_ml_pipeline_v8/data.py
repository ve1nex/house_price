from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV dataset and fail early if the path is wrong."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def prepare_dataframe(df: pd.DataFrame, config) -> pd.DataFrame:
    df = df.copy()

    if config.data.drop_columns:
        df = df.drop(columns=list(config.data.drop_columns), errors="ignore")

    # Semantic missing values
    for column in config.preprocessing.missing_as_none:
        if column in df.columns:
            df[column] = df[column].fillna("None")

    for column in config.preprocessing.missing_as_zero:
        if column in df.columns:
            df[column] = df[column].fillna(0)
    return df


def split_features_target(df: pd.DataFrame, config):
    target = config.data.target
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is missing")

    feature_drop = [target]

    if config.data.id_column is not None and config.data.id_column in df.columns:
        feature_drop.append(config.data.id_column)

    group_column = config.split.group_column
    if group_column is not None and group_column in df.columns:
        feature_drop.append(group_column)

    X = df.drop(columns=list(dict.fromkeys(feature_drop)))
    y = df[target]
    return X, y


def get_groups(df: pd.DataFrame, config):
    """Return group labels for group-based CV, otherwise None."""
    group_column = config.split.group_column
    if group_column is None:
        return None

    if group_column not in df.columns:
        raise KeyError(
            f"Group column '{group_column}' is missing. "
            "Set config.split.group_column correctly or choose a non-group CV strategy."
        )

    return df[group_column].copy()


def build_preprocessor(X: pd.DataFrame, config) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy=config.preprocessing.numeric_imputer))
    ]

    if config.preprocessing.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy=config.preprocessing.categorical_imputer))
    ]

    if config.preprocessing.encode_categorical:
        categorical_steps.append(
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        )

    transformers = []

    if numeric_columns:
        transformers.append(
            ("num", Pipeline(numeric_steps), numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("cat", Pipeline(categorical_steps), categorical_columns)
        )

    if not transformers:
        raise ValueError("No usable feature columns were found")

    return ColumnTransformer(transformers=transformers, remainder="drop")
