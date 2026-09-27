import warnings

import numpy as np
import pandas as pd


def _handle_suspicious(message: str, config) -> None:
    mode = str(config.data_checks.on_suspicious_feature).lower()
    if mode == "raise":
        raise ValueError(message)
    if mode == "warn":
        warnings.warn(message, stacklevel=2)
        return
    raise ValueError("config.data_checks.on_suspicious_feature must be 'warn' or 'raise'")


def check_data_leakage(X: pd.DataFrame, y: pd.Series, config) -> None:
    """Universal first-pass leakage checks.

    This does NOT replace task-specific leakage analysis. Real leakage often depends
    on how and when a feature is created. Keep this block and add project-specific
    checks here when a new task requires them.
    """
    if not config.data_checks.enabled:
        return

    # --- Known leakage columns ---
    # Fill this list from domain knowledge in config when necessary.
    known = set(config.data_checks.known_leakage_columns)
    found = sorted(known.intersection(X.columns))
    if found:
        raise ValueError(f"Known leakage columns are still present in features: {found}")

    # --- Obvious target copy check ---
    # A feature exactly equal to y is almost certainly target leakage.
    y_array = np.asarray(y)
    for column in X.columns:
        series = X[column]
        if len(series) != len(y):
            continue

        try:
            equal_mask = series.notna().to_numpy() & pd.notna(y_array)
            if equal_mask.any() and np.array_equal(
                series.to_numpy()[equal_mask],
                y_array[equal_mask],
            ):
                _handle_suspicious(
                    f"Feature '{column}' is identical to the target on all comparable rows. "
                    "This looks like target leakage.",
                    config,
                )
        except (TypeError, ValueError):
            # Some exotic object columns may not support a meaningful direct comparison.
            pass

    # --- Task-specific leakage checks ---
    # Add ONLY checks that make sense for the current project, for example:
    # - time leakage: a feature contains information from the future;
    # - group leakage: the same user/patient/object appears in train and validation;
    # - post-event columns: a value is only known after the target event happened.
