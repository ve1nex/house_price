import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
from omegaconf import OmegaConf


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_directories(config) -> None:
    """Create all technical output folders used by the pipeline."""
    for path in [
        config.paths.path_to_checkpoints,
        config.paths.path_to_plots,
        config.paths.path_to_logs,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def _directory_has_artifacts(path: Path) -> bool:
    """Return True when an experiment folder already contains something."""
    return path.exists() and any(path.iterdir())


def save_config_snapshot(config) -> None:
    """Save the exact resolved config used by this experiment."""
    if not bool(config.reproducibility.save_config_snapshot):
        return

    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    path = Path(config.paths.path_to_config_snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(resolved, path)


def _safe_package_version(package_name: str):
    """Return an installed package version or None when unavailable."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def save_environment_info(config) -> None:
    """Save Python, OS and important package versions for reproducibility."""
    if not bool(config.reproducibility.save_environment):
        return

    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "omegaconf",
        "matplotlib",
        "shap",
        "optuna",
        "xgboost",
        "lightgbm",
    ]

    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: _safe_package_version(name) for name in packages},
    }

    path = Path(config.paths.path_to_environment)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_experiment(config) -> None:
    """Protect old experiments, create folders and save reproducibility files."""
    experiment_dir = Path(config.paths.path_to_checkpoints)

    if _directory_has_artifacts(experiment_dir):
        if not bool(config.general.overwrite_experiment):
            raise FileExistsError(
                f"Experiment '{config.general.experiment_name}' already exists at {experiment_dir}. "
                "Change config.general.experiment_name or set "
                "config.general.overwrite_experiment=True intentionally."
            )
        shutil.rmtree(experiment_dir)

    ensure_directories(config)
    save_config_snapshot(config)
    save_environment_info(config)


def dataframe_info(df: pd.DataFrame) -> dict:
    """Small serializable schema snapshot without copying the dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 3),
        "columns": [str(column) for column in df.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
    }


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a source-file hash when exact dataset identity is needed."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_dataset_metadata(
    config, raw_info: dict, prepared_info: dict, training_info: dict
) -> None:
    """Save what data/schema was actually used by the experiment."""
    if not bool(config.reproducibility.save_data_info):
        return

    source = Path(config.paths.path_to_train_dataset)
    source_info = {
        "path": str(source),
        "size_bytes": source.stat().st_size if source.exists() else None,
        "sha256": None,
    }

    if bool(config.reproducibility.calculate_dataset_hash) and source.exists():
        source_info["sha256"] = file_sha256(str(source))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": str(config.general.experiment_name),
        "target": str(config.data.target),
        "id_column": None
        if config.data.id_column is None
        else str(config.data.id_column),
        "source": source_info,
        "raw_dataset": raw_info,
        "after_generic_data_stages": prepared_info,
        "actually_used_for_training": training_info,
    }

    path = Path(config.paths.path_to_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_experiment_result(
    config, cv_mean: float, cv_std: float, elapsed_seconds: int
) -> None:
    """Save experiment summary in CSV and/or text format."""
    from estimator_strategy import get_estimator_label

    timestamp = datetime.now().isoformat(timespec="seconds")
    model_label = get_estimator_label(config)

    if config.logging.csv_file:
        path = Path(config.paths.path_to_results_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(
                    [
                        "datetime",
                        "experiment",
                        "task",
                        "model",
                        "metric",
                        "cv_mean",
                        "cv_std",
                        "time_seconds",
                    ]
                )
            writer.writerow(
                [
                    timestamp,
                    config.general.experiment_name,
                    config.general.task,
                    model_label,
                    config.metric.name,
                    round(cv_mean, 6),
                    round(cv_std, 6),
                    elapsed_seconds,
                ]
            )

    if config.logging.txt_file:
        path = Path(config.paths.path_to_results_txt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(
                f"{config.general.experiment_name} | "
                f"{model_label} | "
                f"{config.metric.name}: {cv_mean:.6f} ± {cv_std:.6f} | "
                f"{elapsed_seconds} s\n"
            )


def get_metric(config, y_true, y_pred):
    """Calculate the configured regression metric."""
    name = str(config.metric.name)
    if not hasattr(sklearn.metrics, name):
        raise ValueError(f"Unknown metric: {name}")
    return float(
        getattr(sklearn.metrics, name)(
            y_true, y_pred, **OmegaConf.to_container(config.metric.params, resolve=True)
        )
    )
