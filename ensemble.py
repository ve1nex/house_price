"""One artifact-only Classic/DL ensemble runner; no base model training imports."""
import json
import math
import re
import shutil
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ensemble_config import config as DEFAULT_CONFIG

ALLOWED_META = {
    "LogisticRegression": LogisticRegression, "Ridge": Ridge,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
}
PROBA_METRICS = {"roc_auc_score", "average_precision_score", "log_loss", "brier_score_loss"}


def _safe_name(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe experiment/name: {value!r}")
    return value


def _read_artifact(root, member):
    source = str(member["source"])
    if source not in {"classic", "dl"}:
        raise ValueError(f"Member {member['name']}: source must be 'classic' or 'dl'")
    experiment = _safe_name(member["experiment"])
    directory = Path(root["paths"][f"{source}_checkpoints"]) / experiment
    if not directory.is_dir():
        raise FileNotFoundError(f"Member {member['name']}: experiment directory not found: {directory}")
    paths = {item: directory / item for item in ["oof_predictions.csv", "predictions.csv", "metadata.json"]}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Member {member['name']}: missing {path.name} in {directory}")
    meta = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
    if not meta.get("oof_complete", False):
        raise ValueError(f"Member {member['name']}: OOF is incomplete; train all folds before ensemble")
    if meta.get("test_prediction_stage", "base") != "base":
        raise ValueError(f"Member {member['name']}: test uses self-training while OOF is base; use a consistent evaluation protocol")
    oof = pd.read_csv(paths["oof_predictions.csv"], dtype={"id": str})
    test = pd.read_csv(paths["predictions.csv"], dtype={"id": str})
    for label, df in [("OOF", oof), ("test", test)]:
        if "id" not in df or df["id"].isna().any() or df["id"].duplicated().any():
            raise ValueError(f"Member {member['name']}: {label} IDs are missing or duplicated")
    if "target" not in oof or "fold" not in oof:
        raise ValueError(f"Member {member['name']}: OOF needs target and fold columns")
    if len(oof) != int(meta.get("n_train", -1)):
        raise ValueError(f"Member {member['name']}: OOF row count differs from metadata.n_train")
    task = str(meta.get("task"))
    if task not in {"classification", "regression"}:
        raise ValueError(f"Member {member['name']}: metadata.task is invalid")
    classes = meta.get("classes") if task == "classification" else None
    if task == "classification" and (not isinstance(classes, list) or len(classes) < 2):
        raise ValueError(f"Member {member['name']}: metadata.classes must list ordered classes")
    expected = ["prediction"] if classes is None or len(classes) == 2 else [f"pred_class_{i}" for i in range(len(classes))]
    for label, df in [("OOF", oof), ("test", test)]:
        found = [c for c in df if c == "prediction" or c.startswith("pred_class_")]
        if found != expected:
            raise ValueError(f"Member {member['name']}: {label} prediction columns {found} != {expected}")
        values = df[expected].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Member {member['name']}: {label} contains NaN or infinity")
        if task == "classification" and (np.any(values < -1e-7) or np.any(values > 1 + 1e-7) or
            (len(classes) > 2 and not np.allclose(values.sum(axis=1), 1, atol=1e-4))):
            raise ValueError(f"Member {member['name']}: {label} probabilities are invalid")
    return {"name": str(member["name"]), "member": member, "oof": oof, "test": test,
            "meta": meta, "columns": expected, "task": task, "classes": classes}


def _aligned(config, members):
    first = members[0]
    if not first["meta"].get("id_namespace"):
        raise ValueError("Member metadata must specify a shared ID namespace")
    ref_oof, ref_test = first["oof"], first["test"]
    for member in members[1:]:
        if member["task"] != first["task"] or member["classes"] != first["classes"]:
            raise ValueError(f"Member {member['name']}: task or ordered classes differ")
        if member["meta"].get("id_namespace") != first["meta"].get("id_namespace"):
            raise ValueError(f"Member {member['name']}: ID namespace differs; configure consistent IDs")
        if set(member["oof"].id) != set(ref_oof.id) or set(member["test"].id) != set(ref_test.id):
            raise ValueError(f"Member {member['name']}: OOF/test ID sets differ")
        lookup = member["oof"].set_index("id").loc[ref_oof.id]
        if not np.array_equal(lookup.target.to_numpy(), ref_oof.target.to_numpy()):
            raise ValueError(f"Member {member['name']}: target differs after ID alignment")
        if not np.array_equal(lookup.fold.to_numpy(), ref_oof.fold.to_numpy()):
            raise ValueError(f"Member {member['name']}: fold assignments differ after ID alignment")
        member["oof"] = lookup.reset_index()
        member["test"] = member["test"].set_index("id").loc[ref_test.id].reset_index()
    if ref_oof.fold.isna().any() or ref_oof.fold.nunique() < 2:
        raise ValueError("Ensemble needs at least two complete OOF folds")
    return ref_oof.id.to_numpy(), ref_test.id.to_numpy(), ref_oof.target.to_numpy(), ref_oof.fold.to_numpy(), first["task"], first["classes"]


def _scores(config, y, p, task, classes):
    name = str(config["metric"]["name"])
    params = dict(config["metric"].get("params", {}))
    if not hasattr(sklearn.metrics, name):
        raise ValueError(f"Unknown sklearn metric {name}")
    if name in PROBA_METRICS:
        if task != "classification":
            raise ValueError(f"Metric {name} needs classification")
        v = p if name == "log_loss" or len(classes) > 2 else p[:, 1]
        if name == "log_loss":
            params.setdefault("labels", classes)
        if name == "roc_auc_score" and len(classes) > 2:
            params.setdefault("multi_class", "ovr")
            params.setdefault("labels", classes)
    else:
        v = _labels(p, task, classes)
    return float(getattr(sklearn.metrics, name)(y, v, **params))


def _labels(p, task, classes):
    if task == "regression":
        return np.asarray(p).reshape(-1)
    return np.asarray(classes)[np.argmax(p, axis=1)]


def _matrix(df, cols, task, classes):
    values = df[cols].to_numpy(dtype=float)
    if task == "regression":
        return values
    return np.column_stack((1-values[:, 0], values[:, 0])) if len(classes) == 2 else values


def _combine(stack, weights):
    weights = np.asarray(weights, dtype=float)
    if len(weights) != stack.shape[0] or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Weights must be finite, non-negative, one per member, and sum > 0")
    return np.tensordot(weights / weights.sum(), stack, axes=(0, 0))


def _hard_vote(stack, classes):
    labels = np.argmax(stack, axis=2)
    counts = np.stack([(labels == i).mean(axis=0) for i in range(len(classes))], axis=1)
    return counts


def _meta_model(config, task):
    meta_cfg = config["ensemble"]["meta_model"]
    name = str(meta_cfg[task])
    if name not in ALLOWED_META or (task == "classification") != name.endswith("Classifier") and name != ("LogisticRegression" if task == "classification" else "Ridge"):
        raise ValueError(f"Meta-model {name} is incompatible with {task}")
    params = dict(meta_cfg["params"].get(task, {}))
    return ALLOWED_META[name](**params), name, params


def _meta_output(model, features, task, classes):
    if task == "regression":
        return model.predict(features).reshape(-1, 1)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Classification meta-model must support predict_proba")
    if list(model.classes_) != list(classes):
        raise ValueError("A meta-training fold lacks a class; use different folds")
    return model.predict_proba(features)


def _fit_meta(model, features, y):
    if isinstance(model, LogisticRegression) and model.C == float("inf"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Setting penalty=None will ignore the C and l1_ratio parameters", category=UserWarning, module=r"sklearn\.linear_model\._logistic")
            return model.fit(features, y)
    return model.fit(features, y)


def _optimize_weights(config, stack, y, task, classes, seed):
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Weight optimization needs optuna; install bundle requirements") from error
    if config["ensemble"]["weight_optimization"]["method"] != "optuna":
        raise ValueError("Only Optuna weight optimization is implemented")
    study = optuna.create_study(direction=str(config["metric"]["direction"]),
                                sampler=optuna.samplers.TPESampler(seed=int(seed)))
    def objective(trial):
        values = [trial.suggest_float(f"weight_{i}", 1e-6, 1.0) for i in range(len(stack))]
        return _scores(config, y, _combine(stack, values), task, classes)
    study.optimize(objective, n_trials=int(config["ensemble"]["weight_optimization"]["n_trials"]), n_jobs=1, show_progress_bar=False)
    values = [study.best_params[f"weight_{i}"] for i in range(len(stack))]
    return (np.asarray(values) / sum(values)).tolist()


def _draw_validation(y, p, task, classes, directory):
    if task == "classification":
        fig, ax = plt.subplots(figsize=(7, 6))
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y, _labels(p, task, classes), ax=ax, colorbar=False)
        ax.set_title("OOF ensemble confusion matrix")
    else:
        predicted = p[:, 0]
        residuals = np.asarray(y, float) - predicted
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
        a.scatter(y, predicted, alpha=0.55)
        low, high = min(min(y), min(predicted)), max(max(y), max(predicted))
        a.plot([low, high], [low, high], "--", color="black")
        a.set(xlabel="True", ylabel="Predicted", title="OOF true vs predicted")
        b.scatter(predicted, residuals, alpha=0.55)
        b.axhline(0, color="black", linestyle="--")
        b.set(xlabel="Predicted", ylabel="True − predicted", title="Residuals")
    fig.tight_layout()
    fig.savefig(directory / "validation.png", dpi=140)
    plt.close(fig)


def _draw_cv(scores, metric, directory):
    keys, vals = zip(*scores.items())
    vals = np.asarray(vals, float)
    fig, ax = plt.subplots(figsize=(max(7, len(vals)*1.2), 5))
    bars = ax.bar([str(x) for x in keys], vals)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.axhline(vals.mean(), color="darkred", linestyle="--", label=f"Mean ± std: {vals.mean():.4f} ± {vals.std():.4f}")
    ax.set(xlabel="Fold", ylabel=metric, title="Ensemble CV scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "cv_scores.png", dpi=140)
    plt.close(fig)


def _draw_diagnostics(names, base_scores, overall_score, stack, weights, model, directory):
    panel_weights = weights is not None or (model is not None and hasattr(model, "coef_"))
    fig, axes = plt.subplots(1, 3 if panel_weights else 2, figsize=(15 if panel_weights else 11, 4.5))
    names_scores = list(names) + ["ensemble"]
    axes[0].bar(names_scores, list(base_scores.values()) + [overall_score])
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Base models vs ensemble")
    corr_ax = axes[-1]
    flattened = stack.reshape(len(stack), -1)
    centered = flattened - flattened.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(centered, axis=1)
    safe_lengths = np.where(lengths == 0, 1, lengths)
    corr = (centered @ centered.T) / np.outer(safe_lengths, safe_lengths)
    np.fill_diagonal(corr, 1.0)
    image = corr_ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    corr_ax.set_xticks(range(len(names)), names, rotation=30)
    corr_ax.set_yticks(range(len(names)), names)
    corr_ax.set_title("OOF prediction correlation")
    fig.colorbar(image, ax=corr_ax, shrink=0.8)
    if panel_weights:
        ax = axes[1]
        if weights is not None:
            ax.bar(names, weights)
            ax.set_title("Final model weights")
        else:
            coef = np.asarray(model.coef_)
            vals = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
            ax.bar(range(len(vals)), vals)
            ax.set_title("Meta-model |coefficients|")
            ax.set_xlabel("Meta feature index")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(directory / "ensemble_diagnostics.png", dpi=140)
    plt.close(fig)


def run_ensemble(config=None):
    cfg = DEFAULT_CONFIG if config is None else config
    if not cfg["ensemble"]["enabled"]:
        raise ValueError("Set ensemble.enabled=True in ensemble_config.py")
    started = time.perf_counter()
    preset_name = _safe_name(cfg["ensemble"]["preset"])
    try:
        preset = cfg["ensemble"]["presets"][preset_name]
    except KeyError as error:
        raise ValueError(f"Unknown ensemble preset: {preset_name}") from error
    kind = str(preset["type"])
    if kind not in {"average", "weighted_average", "voting", "stacking"}:
        raise ValueError(f"Unsupported ensemble type: {kind}")
    members_cfg = list(preset["members"])
    if len(members_cfg) < 2 or len({_safe_name(m["name"]) for m in members_cfg}) != len(members_cfg):
        raise ValueError("Ensemble requires at least two uniquely named members")
    members = [_read_artifact(cfg, member) for member in members_cfg]
    oof_ids, test_ids, y, fold, task, classes = _aligned(cfg, members)
    if kind == "voting" and task != "classification":
        raise ValueError("Voting supports classification only")
    if kind == "voting" and cfg["ensemble"]["voting"] == "hard" and str(cfg["metric"]["name"]) in PROBA_METRICS:
        raise ValueError("Hard voting cannot be scored with probability metrics")
    oof_stack = np.stack([_matrix(m["oof"], m["columns"], task, classes) for m in members])
    test_stack = np.stack([_matrix(m["test"], m["columns"], task, classes) for m in members])
    names = [m["name"] for m in members]
    base_scores = {m["name"]: _scores(cfg, y, oof_stack[i], task, classes) for i, m in enumerate(members)}
    weights = None
    meta_model = None
    weight_opt = bool(cfg["optimization"]["enabled"]) and bool(cfg["ensemble"]["weight_optimization"]["enabled"])
    if bool(cfg["ensemble"]["weight_optimization"]["enabled"]) and not bool(cfg["optimization"]["enabled"]):
        weight_opt = False  # master switch ignores child settings
    if weight_opt and kind != "weighted_average":
        raise ValueError("Optuna weight optimization requires weighted_average preset")
    if kind == "weighted_average":
        raw_weights = [m.get("weight") for m in members_cfg]
        if any(w is None for w in raw_weights):
            raise ValueError("weighted_average requires weight for every member")
        _combine(oof_stack, raw_weights)  # validate before any optimization
        weights = (np.asarray(raw_weights, dtype=float) / sum(raw_weights)).tolist()
    folds = np.unique(fold)
    if kind == "stacking":
        features = np.concatenate([oof_stack[i] for i in range(len(members))], axis=1)
        test_features = np.concatenate([test_stack[i] for i in range(len(members))], axis=1)
        oof_pred = np.full((len(y), len(classes) if classes else 1), np.nan)
        for group in folds:
            train_mask, val_mask = fold != group, fold == group
            if task == "classification" and set(np.unique(y[train_mask])) != set(classes):
                raise ValueError(f"Meta training excluding fold {group} lacks a class; use stratified folds")
            model, _, _ = _meta_model(cfg, task)
            _fit_meta(model, features[train_mask], y[train_mask])
            oof_pred[val_mask] = _meta_output(model, features[val_mask], task, classes)
        meta_model, meta_name, meta_params = _meta_model(cfg, task)
        _fit_meta(meta_model, features, y)
        test_pred = _meta_output(meta_model, test_features, task, classes)
    elif weight_opt:
        oof_pred = np.full((len(y), len(classes) if classes else 1), np.nan)
        for group in folds:
            train_mask, val_mask = fold != group, fold == group
            cv_weights = _optimize_weights(cfg, oof_stack[:, train_mask], y[train_mask], task, classes, int(cfg["general"]["seed"])+int(group))
            oof_pred[val_mask] = _combine(oof_stack[:, val_mask], cv_weights)
        weights = _optimize_weights(cfg, oof_stack, y, task, classes, int(cfg["general"]["seed"]))
        test_pred = _combine(test_stack, weights)
    elif kind == "voting" and str(cfg["ensemble"]["voting"]) == "hard":
        oof_pred, test_pred = _hard_vote(oof_stack, classes), _hard_vote(test_stack, classes)
    else:
        active_weights = weights if weights is not None else [1.0] * len(members)
        oof_pred, test_pred = _combine(oof_stack, active_weights), _combine(test_stack, active_weights)
    if task == "classification":
        if not np.allclose(oof_pred.sum(axis=1), 1, atol=1e-5) or not np.allclose(test_pred.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Ensemble probabilities do not sum to 1")
    per_fold = {int(f): _scores(cfg, y[fold == f], oof_pred[fold == f], task, classes) for f in folds}
    overall = _scores(cfg, y, oof_pred, task, classes)
    output_dir = Path(cfg["paths"]["ensembles_root"]) / _safe_name(cfg["general"]["experiment_name"])
    if output_dir.exists() and any(output_dir.iterdir()):
        if not bool(cfg["general"]["overwrite_experiment"]):
            raise FileExistsError(f"Ensemble experiment exists: {output_dir}; choose a new name")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from artifact_format import save_predictions
    out_oof = oof_pred[:, 1] if task == "classification" and len(classes) == 2 else oof_pred[:, 0] if task == "regression" else oof_pred
    out_test = test_pred[:, 1] if task == "classification" and len(classes) == 2 else test_pred[:, 0] if task == "regression" else test_pred
    save_predictions(output_dir / "oof_predictions.csv", oof_ids, out_oof, task, classes, y, fold)
    save_predictions(output_dir / "predictions.csv", test_ids, out_test, task, classes)
    if meta_model is not None:
        joblib.dump(meta_model, output_dir / "meta_model.joblib")
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    metrics = {"metric": cfg["metric"]["name"], "score": overall,
               "fold_scores": {str(k):v for k,v in per_fold.items()},
               "cv_mean": float(np.mean(list(per_fold.values()))), "cv_std": float(np.std(list(per_fold.values()))),
               "base_scores": base_scores}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"experiment": cfg["general"]["experiment_name"], "preset": preset_name, "type": kind,
                "task": task, "classes": classes, "id_namespace": members[0]["meta"].get("id_namespace"),
                "members": [{"name":m["name"], "source":m["member"]["source"],"experiment":m["member"]["experiment"]} for m in members],
                "weights": weights, "weights_optimized": weight_opt,
                "meta_model": None if meta_model is None else {"name":meta_name,"params":meta_params},
                "oof_complete": True, "n_train": len(y), "n_test": len(test_ids),
                "training_time_seconds": time.perf_counter()-started,
                "evaluation": "cross-fitted meta-model/weights" if kind == "stacking" or weight_opt else "OOF blend"}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    if cfg["visualization"]["save_validation_plot"]:
        _draw_validation(y, oof_pred, task, classes, plots)
    if cfg["visualization"]["save_cv_scores"]:
        _draw_cv(per_fold, cfg["metric"]["name"], plots)
    if cfg["visualization"]["save_ensemble_diagnostics"]:
        _draw_diagnostics(names, base_scores, overall, oof_stack, weights if kind == "weighted_average" else None, meta_model, plots)
    return output_dir, metrics


if __name__ == "__main__":
    directory, result = run_ensemble()
    print(f"Saved ensemble: {directory} | {result['metric']}: {result['score']:.6f}")
