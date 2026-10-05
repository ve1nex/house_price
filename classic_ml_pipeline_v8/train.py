import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from artifact_io import save_predictions, update_metadata
from data import build_preprocessor
from estimator_strategy import get_estimator
from utils import get_metric, save_experiment_result
from visualization import (
    save_cv_scores,
    save_feature_importance,
    save_shap_summary,
    save_validation_plot,
)


def get_cv(config):
    """Build the KFold splitter used for this regression task."""
    if str(config.split.strategy) != "KFold":
        raise ValueError("House Prices regression uses KFold")
    return KFold(
        n_splits=int(config.split.n_splits),
        shuffle=bool(config.split.shuffle),
        random_state=int(config.general.seed) if config.split.shuffle else None,
    )


def build_training_pipeline(X, config):
    """Fit learned preprocessing separately inside every outer training fold."""
    return Pipeline(
        [
            ("preprocessing", build_preprocessor(X, config)),
            ("model", get_estimator(config)),
        ]
    )


def _transform_target(y, config):
    """Transform the target once; validation and OOF remain in this space."""
    transform = str(config.data.target_transform)
    if transform == "none":
        return y
    if transform != "log1p" or (np.asarray(y) < 0).any():
        raise ValueError(
            "target_transform must be none or log1p with non-negative targets"
        )
    return pd.Series(np.log1p(np.asarray(y, dtype=float)), index=y.index, name=y.name)


def evaluate_cv_score(config, X, y, folds_to_use=None):
    """Evaluate selected folds for Optuna without writing model artifacts."""
    y = _transform_target(y, config)
    requested = set(
        int(f)
        for f in (
            folds_to_use if folds_to_use is not None else config.split.folds_to_train
        )
    )
    scores = []
    for fold, (tr, va) in enumerate(get_cv(config).split(X)):
        if fold in requested:
            model = build_training_pipeline(X.iloc[tr], config)
            model.fit(X.iloc[tr], y.iloc[tr])
            scores.append(get_metric(config, y.iloc[va], model.predict(X.iloc[va])))
    if not requested or len(scores) != len(requested):
        raise ValueError("Requested tuning folds do not exist")
    return float(np.mean(scores))


def train(config, X, y, ids=None):
    """Save OOF predictions, CV diagnostics, and an optional model fitted on all rows."""
    y = _transform_target(y, config)
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    if len(ids) != len(y) or pd.Index(ids).has_duplicates or pd.isna(ids).any():
        raise ValueError("Training IDs must be unique, non-null, and match labels")
    requested = set(int(f) for f in config.split.folds_to_train)
    predictions = np.full(len(y), np.nan)
    assignments = np.full(len(y), -1, dtype=int)
    fold_scores = []
    started = time.perf_counter()
    for fold, (tr, va) in enumerate(get_cv(config).split(X)):
        if fold not in requested:
            continue
        model = build_training_pipeline(X.iloc[tr], config)
        model.fit(X.iloc[tr], y.iloc[tr])
        predictions[va] = model.predict(X.iloc[va])
        assignments[va] = fold
        score = get_metric(config, y.iloc[va], predictions[va])
        fold_scores.append((fold, score))
        if config.logging.prints:
            print(f"Fold {fold} | {config.metric.name}: {score:.5f}")
    if not requested or len(fold_scores) != len(requested):
        raise ValueError("Requested folds do not exist")
    selected = assignments >= 0
    if config.training.save_oof_predictions:
        save_predictions(
            config.paths.path_to_oof,
            ids[selected],
            predictions[selected],
            "regression",
            target=np.asarray(y)[selected],
            folds=assignments[selected],
        )
    save_validation_plot(np.asarray(y)[selected], predictions[selected], config)
    save_cv_scores(fold_scores, config)
    final_model = None
    if config.training.train_final_model:
        final_model = build_training_pipeline(X, config)
        final_model.fit(X, y)
        joblib.dump(final_model, config.paths.path_to_final_model)
        save_feature_importance(final_model, X, config)
        save_shap_summary(final_model, X, config)
        if config.training.predict_test_after_fit:
            from predict import inference

            inference(config)
    scores = [score for _, score in fold_scores]
    mean, std = float(np.mean(scores)), float(np.std(scores))
    elapsed = time.perf_counter() - started
    update_metadata(
        config.paths.path_to_metadata,
        task="regression",
        classes=None,
        id_namespace=str(config.data.id_namespace),
        metric=str(config.metric.name),
        metric_direction=str(config.metric.direction),
        fold_scores={str(f): s for f, s in fold_scores},
        cv_mean=mean,
        cv_std=std,
        oof_complete=bool(selected.all()),
        n_oof=int(selected.sum()),
        n_train=len(y),
        training_time_seconds=elapsed,
    )
    save_experiment_result(config, mean, std, elapsed)
    if config.logging.prints:
        print(f"CV {config.metric.name}: {mean:.4f} ± {std:.4f}")
    return final_model, scores
