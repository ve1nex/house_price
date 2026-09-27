"""Stable prediction artifacts shared by Classic and DL (no model dependencies)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _table(ids, values, task, classes=None, target=None, folds=None):
    ids = np.asarray(ids)
    values = np.asarray(values)
    if ids.ndim != 1 or len(ids) != len(values) or pd.isna(ids).any() or pd.Index(ids).has_duplicates:
        raise ValueError("Artifact IDs must be non-null, unique, and match predictions length")
    if not np.isfinite(values.astype(float)).all():
        raise ValueError("Predictions contain NaN or infinity")
    df = pd.DataFrame({"id": ids})
    if target is not None:
        if len(target) != len(ids) or pd.isna(target).any():
            raise ValueError("Target length or values are invalid")
        df["target"] = np.asarray(target)
    if task == "classification" and classes is not None and len(classes) > 2:
        if values.shape != (len(ids), len(classes)):
            raise ValueError("Multiclass predictions must be a [rows, classes] probability matrix")
        for index in range(len(classes)):
            df[f"pred_class_{index}"] = values[:, index]
    else:
        if values.ndim != 1:
            raise ValueError("Binary probability or regression prediction must be one-dimensional")
        df["prediction"] = values
    if task == "classification":
        probabilities = df.filter(regex="^(prediction|pred_class_)").to_numpy(dtype=float)
        if np.any((probabilities < -1e-7) | (probabilities > 1 + 1e-7)):
            raise ValueError("Classification probabilities must lie in [0, 1]")
        if probabilities.shape[1] > 1 and not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
            raise ValueError("Multiclass probability rows must sum to 1")
    if folds is not None:
        if target is None or len(folds) != len(ids):
            raise ValueError("OOF fold labels must match IDs")
        df["fold"] = np.asarray(folds, dtype=int)
    return df


def save_predictions(path, ids, values, task, classes=None, target=None, folds=None):
    df = _table(ids, values, task, classes, target, folds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def update_metadata(path, **fields):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
