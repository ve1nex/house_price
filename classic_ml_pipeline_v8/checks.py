import warnings

import numpy as np
import pandas as pd


def _handle_suspicious(message, config):
    """Warn or fail according to the configured leakage check policy."""
    mode = str(config.data_checks.on_suspicious_feature)
    if mode == "raise":
        raise ValueError(message)
    if mode == "warn":
        warnings.warn(message, stacklevel=2)
        return
    raise ValueError("on_suspicious_feature must be warn or raise")


def check_data_leakage(X, y, config):
    """Reject known leakage columns and identify features that copy the raw target."""
    if not config.data_checks.enabled:
        return
    if str(config.data.target) in X:
        raise ValueError("Target column leaked into features")
    found = sorted(
        set(config.data_checks.known_leakage_columns).intersection(X.columns)
    )
    if found:
        raise ValueError(f"Known leakage features: {found}")
    target = np.asarray(y)
    for column in X:
        values = X[column]
        comparable = values.notna().to_numpy() & pd.notna(target)
        if comparable.any() and np.array_equal(
            values.to_numpy()[comparable], target[comparable]
        ):
            _handle_suspicious(
                f"Feature {column!r} is identical to the target on comparable rows",
                config,
            )
