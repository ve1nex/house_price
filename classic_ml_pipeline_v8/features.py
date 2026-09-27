import pandas as pd


def feature_engineering(df: pd.DataFrame, config) -> pd.DataFrame:
    """Project-specific feature hook. Add train/test-safe transformations here."""
    if not bool(config.feature_engineering.enabled):
        return df
    # Replace this exception with your project's feature logic.
    raise NotImplementedError("Implement feature_engineering in features.py before enabling it")
