"""House Prices table preparation; learned transforms are fitted inside each fold."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_AS_NONE = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]
MISSING_AS_ZERO = [
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
]


def prepare_dataframe(df, config):
    """Apply fixed missing-value rules and the retained features without fitting statistics."""
    df = df.drop(columns=list(config.data.drop_columns), errors="ignore").copy()
    for column in MISSING_AS_NONE:
        if column in df:
            df[column] = df[column].fillna("None")
    for column in MISSING_AS_ZERO:
        if column in df:
            df[column] = df[column].fillna(0)
    df["MSSubClassCat"] = df["MSSubClass"].astype(str)
    df["Spaciousness"] = (df["1stFlrSF"] + df["2ndFlrSF"]) / df["TotRmsAbvGrd"].replace(
        0, np.nan
    )
    return df


def load_raw_data(config, train=True):
    """Read a raw table and preserve sample IDs after the fixed outlier exclusions."""
    path = (
        config.paths.path_to_train_dataset
        if train
        else config.paths.path_to_test_dataset
    )
    df = pd.read_csv(path)
    if train:
        df = df.loc[
            ~df[str(config.data.id_column)].isin(config.data.drop_train_ids)
        ].reset_index(drop=True)
    ids = df[str(config.data.id_column)].to_numpy()
    target = df[str(config.data.target)].to_numpy(dtype=np.float32) if train else None
    if train and str(config.data.target_transform) == "log1p":
        target = np.log1p(target)
    elif train and str(config.data.target_transform) != "none":
        raise ValueError("Unknown target_transform")
    X = df.drop(
        columns=[str(config.data.id_column), str(config.data.target)], errors="ignore"
    )
    return prepare_dataframe(X, config), target, ids


def build_preprocessor(X):
    """Create a dense numeric/one-hot transform to fit on an outer training fold."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def prepare_fold(features, train_idx, val_idx):
    """Fit only training rows, then transform both parts with the same learned parameters."""
    preprocessor = build_preprocessor(features)
    train = preprocessor.fit_transform(features.iloc[train_idx])
    valid = preprocessor.transform(features.iloc[val_idx])
    return (
        np.asarray(train, dtype=np.float32),
        np.asarray(valid, dtype=np.float32),
        preprocessor,
    )
