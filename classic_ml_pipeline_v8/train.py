import time
import json
import warnings
import sys
try:
    import resource
except ImportError:  # Windows
    resource = None
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from data import build_preprocessor
from artifact_io import save_predictions, update_metadata
from estimator_strategy import get_estimator, get_estimator_label
from notifier import send_telegram
from postprocessing import postprocess_predictions
from tracking import log_metrics, log_summary
from utils import PROBABILITY_METRICS, get_metric, save_experiment_result
from visualization import save_feature_importance, save_shap_summary, save_validation_plot, save_cv_scores


def get_cv(config):
    """Create a CV splitter selected in config."""
    strategy = str(config.split.strategy)
    n_splits = int(config.split.n_splits)
    shuffle = bool(config.split.shuffle)
    seed = int(config.general.seed)

    if strategy == "KFold":
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )

    if strategy == "StratifiedKFold":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )

    if strategy == "GroupKFold":
        # GroupKFold itself does not shuffle groups in widely used sklearn versions.
        return GroupKFold(n_splits=n_splits)

    if strategy == "StratifiedGroupKFold":
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )

    raise ValueError(
        "Unknown config.split.strategy. Use one of: "
        "KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold"
    )


def get_split_iterator(cv, X, y, groups, config):
    strategy = str(config.split.strategy)

    if strategy == "KFold":
        return cv.split(X)

    if strategy == "StratifiedKFold":
        return cv.split(X, y)

    if strategy == "GroupKFold":
        if groups is None:
            raise ValueError("GroupKFold requires config.split.group_column")
        return cv.split(X, y, groups=groups)

    if strategy == "StratifiedGroupKFold":
        if groups is None:
            raise ValueError("StratifiedGroupKFold requires config.split.group_column")
        return cv.split(X, y, groups=groups)

    raise ValueError(f"Unsupported CV strategy: {strategy}")


def build_training_pipeline(X, config) -> Pipeline:
    return Pipeline([
        ("preprocessing", build_preprocessor(X, config)),
        ("model", get_estimator(config)),
    ])


def _check_optimization(config):
    if not bool(config.optimization.enabled):
        return
    for section, feature in [("prediction", "calibration"), ("features", "feature_selection")]:
        if bool(config.optimization[section][feature].enabled):
            raise NotImplementedError(f"optimization.{section}.{feature} is a placeholder; add a task-specific implementation")
    if bool(config.optimization.prediction.threshold_tuning.enabled) and (
        str(config.general.task) != "classification" or int(config.general.num_classes) != 2
    ):
        raise ValueError("Threshold tuning supports binary classification only")


def _fit_pipeline(pipeline, X_train, y_train, config, X_val=None, y_val=None, final_iterations=None):
    model = pipeline.named_steps["model"]
    name = str(config.model.name)
    if final_iterations is not None:
        key = "iterations" if name.startswith("CatBoost") else "n_estimators"
        if key in model.get_params():
            model.set_params(**{key: int(final_iterations)})
    early = bool(config.optimization.enabled) and bool(config.optimization.training_control.boosting_early_stopping.enabled)
    if early and X_val is not None:
        if not name.startswith(("XGB", "LGBM", "CatBoost")) or bool(config.estimator_strategy.enabled):
            raise ValueError("Boosting early stopping needs one XGB/LGBM/CatBoost estimator")
        pre = pipeline.named_steps["preprocessing"]
        xt = pre.fit_transform(X_train, y_train)
        xv = pre.transform(X_val)
        rounds = int(config.optimization.training_control.boosting_early_stopping.rounds)
        if rounds <= 0:
            raise ValueError("Early stopping rounds must be positive")
        if name.startswith("XGB"):
            model.set_params(early_stopping_rounds=rounds)
            model.fit(xt, y_train, eval_set=[(xv, y_val)], verbose=False)
        elif name.startswith("LGBM"):
            import lightgbm as lgb
            model.fit(xt, y_train, eval_X=xv, eval_y=y_val, callbacks=[lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(period=0)])
        else:
            model.fit(xt, y_train, eval_set=(xv, y_val), early_stopping_rounds=rounds, use_best_model=True, verbose=False)
        return pipeline
    if name == "LogisticRegression" and float(config.models.LogisticRegression.C) == float("inf"):
        # sklearn 1.8 emits an upstream spurious warning for the documented C=inf API.
        # Suppress only that exact warning; other convergence/deprecation warnings surface.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Setting penalty=None will ignore the C and l1_ratio parameters", category=UserWarning, module=r"sklearn\.linear_model\._logistic")
            pipeline.fit(X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline


def _predict(pipeline, X, config, classes):
    labels = pipeline.predict(X)
    if str(config.general.task) != "classification":
        return np.asarray(labels), np.asarray(labels, dtype=float)
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Classification artifact requires predict_proba; choose a probability estimator or soft voting")
    actual = list(pipeline.classes_)
    if actual != list(classes):
        raise ValueError(f"Fold classes {actual} differ from training classes {classes}; adjust folds")
    probabilities = np.asarray(pipeline.predict_proba(X), dtype=float)
    return np.asarray(labels), probabilities


def _select_threshold(config, targets, probs, classes):
    params = config.optimization.prediction.threshold_tuning
    steps = int(params.n_steps)
    if steps < 3:
        raise ValueError("threshold_tuning.n_steps must be >= 3")
    values = np.linspace(0.0, 1.0, steps)
    scores = [get_metric(config, targets, np.where(probs >= threshold, classes[1], classes[0])) for threshold in values]
    index = int(np.argmax(scores) if str(config.metric.direction) == "maximize" else np.argmin(scores))
    return float(values[index])


def evaluate_cv_score(config, X, y, groups=None, folds_to_use=None):
    """Optuna evaluation without artifacts; same fold-safe preprocessing as train."""
    _check_optimization(config)
    classes = list(np.unique(y)) if str(config.general.task) == "classification" else None
    requested = set(int(f) for f in (folds_to_use if folds_to_use is not None else config.split.folds_to_train))
    scores = []
    for fold, (tr, va) in enumerate(get_split_iterator(get_cv(config), X, y, groups, config)):
        if fold not in requested:
            continue
        fitted = _fit_pipeline(build_training_pipeline(X.iloc[tr], config), X.iloc[tr], y.iloc[tr], config, X.iloc[va], y.iloc[va])
        labels, probs = _predict(fitted, X.iloc[va], config, classes)
        scores.append(get_metric(config, y.iloc[va], labels, probs, classes))
    if not scores or len(scores) != len(requested):
        raise ValueError("Requested tuning folds do not exist")
    return float(np.mean(scores))


def train(config, X, y, groups=None, ids=None):
    """CV, compatible OOF artifacts, optional final fit and test inference."""
    _check_optimization(config)
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    if len(ids) != len(y) or pd.Index(ids).has_duplicates or pd.isna(ids).any():
        raise ValueError("Training IDs must be unique, non-null and match labels")
    classes = list(np.unique(y)) if str(config.general.task) == "classification" else None
    requested = set(int(fold) for fold in config.split.folds_to_train)
    scores, fold_scores = [], []
    n = len(y)
    oof = np.full((n, len(classes)) if classes is not None else n, np.nan, dtype=float)
    fold_assignment = np.full(n, -1, dtype=int)
    best_iterations = []
    start_time = time.perf_counter()
    send_telegram(f"{config.general.experiment_name} started | {get_estimator_label(config)}", config)
    for fold, (tr, va) in enumerate(get_split_iterator(get_cv(config), X, y, groups, config)):
        if fold not in requested:
            continue
        pipeline = _fit_pipeline(build_training_pipeline(X.iloc[tr], config), X.iloc[tr], y.iloc[tr], config, X.iloc[va], y.iloc[va])
        labels, probabilities = _predict(pipeline, X.iloc[va], config, classes)
        score = get_metric(config, y.iloc[va], labels, probabilities, classes)
        scores.append(score)
        fold_scores.append((fold, score))
        oof[va] = probabilities
        fold_assignment[va] = fold
        model = pipeline.named_steps["model"]
        best = getattr(model, "best_iteration_", None)
        if best is None:
            best = getattr(model, "best_iteration", None)
            if best is not None:
                best = int(best) + 1  # XGBoost index starts at 0
        if best is None and hasattr(model, "get_best_iteration"):
            best = model.get_best_iteration()
            if best is not None:
                best = int(best) + 1  # CatBoost index starts at 0
        if best is not None and int(best) > 0:
            best_iterations.append(int(best))
        if bool(config.training.save_fold_models):
            path = Path(config.paths.path_to_fold_models) / f"fold_{fold}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, path)
        if bool(config.logging.prints):
            print(f"Fold {fold} | {config.metric.name}: {score:.5f}")
        log_metrics(config, {"fold": fold, f"fold/{config.metric.name}": score}, step=len(scores)-1)
    if not scores or len(scores) != len(requested):
        raise ValueError("No results for one or more requested folds")
    mask = fold_assignment >= 0
    complete = bool(mask.all())
    threshold = None
    honest_tuned_score = None
    labels_oof = np.asarray(y)[mask] if classes is None else np.asarray(classes)[np.argmax(oof[mask], axis=1)]
    if classes is not None and len(classes) == 2:
        labels_oof = np.where(oof[mask, 1] >= 0.5, classes[1], classes[0])
    if bool(config.optimization.enabled) and bool(config.optimization.prediction.threshold_tuning.enabled):
        if str(config.metric.name) in PROBABILITY_METRICS:
            raise ValueError("Threshold tuning requires a label-based metric")
        fold_labels = fold_assignment[mask]
        tuned = np.empty(len(fold_labels), dtype=np.asarray(y).dtype)
        if len(np.unique(fold_labels)) < 2:
            raise ValueError("Cross-fitted threshold tuning needs at least two trained folds")
        for fold in np.unique(fold_labels):
            other = fold_labels != fold
            t = _select_threshold(config, np.asarray(y)[mask][other], oof[mask, 1][other], classes)
            tuned[~other] = np.where(oof[mask, 1][~other] >= t, classes[1], classes[0])
        honest_tuned_score = get_metric(config, np.asarray(y)[mask], tuned)
        labels_oof = tuned
        threshold = _select_threshold(config, np.asarray(y)[mask], oof[mask, 1], classes)
        threshold_path = Path(config.paths.path_to_checkpoints) / "threshold.json"
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text(json.dumps({"threshold": threshold, "crossfit_score": honest_tuned_score}), encoding="utf-8")
    if bool(config.training.save_oof_predictions):
        values = oof[mask]
        if classes is not None and len(classes) == 2:
            values = values[:, 1]
        save_predictions(config.paths.path_to_oof, ids[mask], values, str(config.general.task), classes, np.asarray(y)[mask], fold_assignment[mask])
    save_validation_plot(np.asarray(y)[mask], labels_oof, config)
    save_cv_scores(fold_scores, config)
    final_pipeline = None
    if bool(config.training.train_final_model):
        iterations = int(np.median(best_iterations)) if best_iterations else None
        final_pipeline = _fit_pipeline(build_training_pipeline(X, config), X, y, config, final_iterations=iterations)
        Path(config.paths.path_to_final_model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipeline, config.paths.path_to_final_model)
        save_feature_importance(final_pipeline, X, config)
        save_shap_summary(final_pipeline, X, config)
        if bool(config.training.predict_test_after_fit) and Path(config.paths.path_to_test_dataset).exists():
            from predict import inference
            inference(config)
    elapsed = time.perf_counter() - start_time
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
    peak_mb = peak_rss / (1024 * 1024 if sys.platform == "darwin" else 1024) if peak_rss is not None else None
    cv_mean, cv_std = float(np.mean(scores)), float(np.std(scores))
    update_metadata(config.paths.path_to_metadata, task=str(config.general.task), classes=None if classes is None else [v.item() if hasattr(v,"item") else v for v in classes], id_namespace=str(config.data.id_namespace), metric=str(config.metric.name), metric_direction=str(config.metric.direction), fold_scores={str(f): v for f,v in fold_scores}, cv_mean=cv_mean, cv_std=cv_std, oof_complete=complete, n_oof=int(mask.sum()), n_train=n, threshold=threshold, threshold_crossfit_score=honest_tuned_score, training_time_seconds=elapsed, peak_process_rss_mb=peak_mb, final_boosting_iterations=iterations if final_pipeline is not None else None)
    save_experiment_result(config, cv_mean, cv_std, int(elapsed))
    log_summary(config, {"cv_mean": cv_mean, "cv_std": cv_std, "time_seconds": elapsed})
    if bool(config.logging.prints):
        print(f"CV {config.metric.name}: {cv_mean:.4f} ± {cv_std:.4f} | {elapsed:.1f}s")
    send_telegram(f"{config.general.experiment_name} finished | CV {cv_mean:.4f} ± {cv_std:.4f}", config)
    return final_pipeline, scores
