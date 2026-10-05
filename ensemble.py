"""Combine saved House Prices OOF/test predictions without retraining base models."""

import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

from ensemble_config import config as DEFAULT_CONFIG


def _safe_name(value):
    """Limit experiment names to one filename component."""
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Invalid experiment name: {value!r}")
    return value


def _read_artifact(config, member):
    """Read and validate a complete regression OOF/test artifact pair."""
    source = str(member["source"])
    if source not in {"classic", "dl"}:
        raise ValueError("Member source must be classic or dl")
    directory = Path(config["paths"][f"{source}_checkpoints"]) / _safe_name(
        member["experiment"]
    )
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("task") != "regression" or not metadata.get("oof_complete"):
        raise ValueError(
            f"Member {member['name']}: complete regression OOF is required"
        )
    oof = pd.read_csv(directory / "oof_predictions.csv")
    test = pd.read_csv(directory / "predictions.csv").rename(
        columns={"Id": "id", "SalePrice": "prediction"}
    )
    if not {"id", "target", "prediction", "fold"}.issubset(oof):
        raise ValueError("OOF needs id, target, prediction, and fold")
    if not {"id", "prediction"}.issubset(test):
        raise ValueError("Test predictions need sample IDs and prices")
    for label, frame in [("OOF", oof), ("test", test)]:
        if frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError(f"{label} IDs are missing or duplicated")
        if not np.isfinite(frame.prediction).all():
            raise ValueError(f"{label} predictions contain NaN or infinity")
    if not np.isfinite(oof.target).all() or oof.fold.isna().any():
        raise ValueError("OOF targets/folds are incomplete")
    if len(oof) != int(metadata.get("n_train", -1)):
        raise ValueError("OOF row count differs from metadata.n_train")
    if str(config["ensemble"]["target_transform"]) == "log1p":
        if (test.prediction < 0).any():
            raise ValueError("Sale prices must be non-negative")
        test["prediction"] = np.log1p(test.prediction.to_numpy(dtype=float))
    elif str(config["ensemble"]["target_transform"]) != "none":
        raise ValueError("Unknown target_transform")
    return {
        "name": str(member["name"]),
        "member": member,
        "metadata": metadata,
        "oof": oof,
        "test": test,
    }


def _aligned(members):
    """Align by sample ID and reject different targets or validation assignments."""
    first = members[0]
    oof, test = first["oof"], first["test"]
    namespace = first["metadata"].get("id_namespace")
    if not namespace:
        raise ValueError("An ID namespace is required")
    for member in members[1:]:
        if member["metadata"].get("id_namespace") != namespace:
            raise ValueError("Member ID namespaces differ")
        if set(member["oof"].id) != set(oof.id) or set(member["test"].id) != set(
            test.id
        ):
            raise ValueError("Member ID sets differ")
        aligned = member["oof"].set_index("id").loc[oof.id].reset_index()
        if not np.allclose(aligned.target, oof.target, rtol=1e-6, atol=1e-6):
            raise ValueError("Member target differs after ID alignment")
        if not np.array_equal(aligned.fold, oof.fold):
            raise ValueError("Member fold assignments differ after ID alignment")
        member["oof"] = aligned
        member["test"] = member["test"].set_index("id").loc[test.id].reset_index()
    if oof.fold.nunique() < 2:
        raise ValueError("At least two complete validation folds are required")
    return oof, test


def _rmse(target, prediction):
    """Calculate RMSE in the configured target space."""
    return float(np.sqrt(np.mean((np.asarray(target) - np.asarray(prediction)) ** 2)))


def _plots(config, directory, y, prediction, folds, base, names, metrics, meta_model):
    """Save regression diagnostics from existing OOF predictions."""
    directory.mkdir(parents=True, exist_ok=True)
    settings = config["visualization"]
    if settings["save_validation_plot"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(y, prediction, s=9, alpha=0.4)
        limits = [min(y.min(), prediction.min()), max(y.max(), prediction.max())]
        axes[0].plot(limits, limits, "--", color="black")
        axes[0].set(xlabel="True log1p price", ylabel="OOF prediction")
        axes[1].scatter(prediction, y - prediction, s=9, alpha=0.4)
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].set(xlabel="Prediction", ylabel="Residual")
        fig.tight_layout()
        fig.savefig(directory / "validation.png", dpi=150)
        plt.close(fig)
    if settings["save_cv_scores"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(map(str, folds)), [metrics["fold_scores"][str(f)] for f in folds])
        ax.axhline(metrics["cv_mean"], color="black", linestyle="--")
        ax.set(
            xlabel="Fold",
            ylabel="RMSE",
            title=f"CV {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}",
        )
        fig.tight_layout()
        fig.savefig(directory / "cv_scores.png", dpi=150)
        plt.close(fig)
    if settings["save_ensemble_diagnostics"]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].bar(
            names + ["ensemble"],
            list(metrics["base_scores"].values()) + [metrics["score"]],
        )
        axes[0, 0].set(ylabel="Global OOF RMSE", title="Base models and ensemble")
        im = axes[0, 1].imshow(np.corrcoef(base.T), vmin=-1, vmax=1, cmap="coolwarm")
        axes[0, 1].set(
            xticks=range(len(names)),
            yticks=range(len(names)),
            xticklabels=names,
            yticklabels=names,
            title="OOF prediction correlation",
        )
        fig.colorbar(im, ax=axes[0, 1])
        values = (
            np.asarray(meta_model.coef_).reshape(-1)
            if meta_model is not None
            else np.full(len(names), 1 / len(names))
        )
        axes[1, 0].bar(names, values)
        axes[1, 0].set(
            title="Meta-model coefficients"
            if meta_model is not None
            else "Averaging weights"
        )
        axes[1, 1].boxplot(
            [y - base[:, i] for i in range(len(names))] + [y - prediction],
            tick_labels=names + ["ensemble"],
            showfliers=False,
        )
        axes[1, 1].set(title="OOF residuals", ylabel="True minus prediction")
        fig.tight_layout()
        fig.savefig(directory / "ensemble_diagnostics.png", dpi=150)
        plt.close(fig)


def run_ensemble(config=None):
    """Rebuild an average, or fit the previously used OOF Ridge stacking method."""
    cfg = DEFAULT_CONFIG if config is None else config
    preset = cfg["ensemble"]["presets"][cfg["ensemble"]["preset"]]
    kind = str(preset["type"])
    if kind not in {"average", "stacking"}:
        raise ValueError("House Prices artifact ensemble supports average and stacking")
    member_specs = list(preset["members"])
    if len(member_specs) < 2 or len({m["name"] for m in member_specs}) != len(
        member_specs
    ):
        raise ValueError("At least two uniquely named members are required")
    members = [_read_artifact(cfg, m) for m in member_specs]
    reference, reference_test = _aligned(members)
    y, fold = reference.target.to_numpy(), reference.fold.to_numpy()
    base = np.column_stack([m["oof"].prediction for m in members])
    test_base = np.column_stack([m["test"].prediction for m in members])
    folds = np.unique(fold)
    meta_model = None
    if kind == "average":
        prediction, test_prediction = base.mean(axis=1), test_base.mean(axis=1)
    else:
        params = cfg["ensemble"]["meta_model"]
        if params["name"] != "Ridge":
            raise ValueError("The saved artifact stacking method uses Ridge")
        prediction = np.full(len(y), np.nan)
        for f in folds:
            model = Ridge(**params["params"])
            model.fit(base[fold != f], y[fold != f])
            prediction[fold == f] = model.predict(base[fold == f])
        meta_model = Ridge(**params["params"]).fit(base, y)
        test_prediction = meta_model.predict(test_base)
    per_fold = {str(f): _rmse(y[fold == f], prediction[fold == f]) for f in folds}
    metrics = {
        "metric": "root_mean_squared_error",
        "score": _rmse(y, prediction),
        "fold_scores": per_fold,
        "cv_mean": float(np.mean(list(per_fold.values()))),
        "cv_std": float(np.std(list(per_fold.values()))),
        "base_scores": {m["name"]: _rmse(y, base[:, i]) for i, m in enumerate(members)},
    }
    directory = Path(cfg["paths"]["ensembles_root"]) / _safe_name(
        cfg["general"]["experiment_name"]
    )
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Experiment already exists: {directory}; select a new name"
        )
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"id": reference.id, "target": y, "prediction": prediction, "fold": fold}
    ).to_csv(directory / "oof_predictions.csv", index=False)
    prices = (
        np.expm1(test_prediction)
        if cfg["ensemble"]["target_transform"] == "log1p"
        else test_prediction
    )
    pd.DataFrame({"Id": reference_test.id, "SalePrice": prices}).to_csv(
        directory / "predictions.csv", index=False
    )
    if meta_model is not None:
        joblib.dump(meta_model, directory / "meta_model.joblib")
    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "task": "regression",
                "n_train": len(y),
                "n_test": len(prices),
                "oof_complete": True,
                "id_namespace": members[0]["metadata"]["id_namespace"],
                "members": member_specs,
                "type": kind,
                "evaluation": "OOF blend"
                if kind == "average"
                else "OOF meta-model cross-fitting",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    _plots(
        cfg,
        directory / "plots",
        y,
        prediction,
        folds,
        base,
        [m["name"] for m in members],
        metrics,
        meta_model,
    )
    return directory, metrics


if __name__ == "__main__":
    directory, metrics = run_ensemble()
    print(
        f"Saved ensemble: {directory} | CV: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"
    )
