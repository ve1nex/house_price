import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_predictions(
    path, ids, values, task="regression", classes=None, target=None, folds=None
):
    """Save finite regression predictions together with unique IDs and optional OOF labels."""
    if task != "regression":
        raise ValueError("House Prices artifacts contain regression predictions")
    ids, values = np.asarray(ids), np.asarray(values).reshape(-1)
    if len(ids) != len(values) or pd.Index(ids).has_duplicates or pd.isna(ids).any():
        raise ValueError("Prediction IDs must be unique, non-null, and match row count")
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinity")
    frame = pd.DataFrame({"id": ids, "prediction": values})
    if target is not None:
        if len(target) != len(ids) or not np.isfinite(target).all():
            raise ValueError("Invalid OOF targets")
        frame.insert(1, "target", np.asarray(target))
    if folds is not None:
        if target is None or len(folds) != len(ids):
            raise ValueError("Invalid OOF fold assignment")
        frame["fold"] = np.asarray(folds, dtype=int)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def update_metadata(path, **fields):
    """Merge metric fields into the existing experiment metadata."""
    path = Path(path)
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
