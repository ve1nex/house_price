import numpy as np
import pandas as pd


def feature_engineering(df: pd.DataFrame, config) -> pd.DataFrame:
    """Add the two features retained after the House Prices experiments."""
    if not bool(config.feature_engineering.enabled):
        return df
    df = df.copy()
    df["MSSubClassCat"] = df["MSSubClass"].astype(str)
    df["Spaciousness"] = (df["1stFlrSF"] + df["2ndFlrSF"]) / df["TotRmsAbvGrd"].replace(
        0, np.nan
    )
    return df
