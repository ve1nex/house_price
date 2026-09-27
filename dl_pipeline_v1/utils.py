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
import torch
from omegaconf import OmegaConf


def resolve_device(config):
    requested = str(config.training.device)
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed, deterministic=True):
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False  # no hidden autotuning speed technique
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)


def ensure_directories(config):
    for path in [
        config.paths.path_to_checkpoints,
        config.paths.path_to_fold_checkpoints,
        config.paths.path_to_plots,
        config.paths.path_to_exports,
        config.paths.path_to_self_training,
        config.paths.path_to_logs,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def _safe_package_version(package_name):
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def save_config_snapshot(config):
    if not bool(config.reproducibility.save_config_snapshot):
        return
    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    path = Path(config.paths.path_to_config_snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(resolved, path)


def save_environment_info(config):
    if not bool(config.reproducibility.save_environment):
        return

    packages = [
        "torch",
        "numpy",
        "scikit-learn",
        "omegaconf",
        "matplotlib",
        "requests",
        "wandb",
        "tensorboard",
        "onnx",
    ]
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "packages": {name: _safe_package_version(name) for name in packages},
    }
    path = Path(config.paths.path_to_environment)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_experiment(config):
    experiment_dir = Path(config.paths.path_to_checkpoints)
    has_artifacts = experiment_dir.exists() and any(experiment_dir.iterdir())

    # Resume is an intentional continuation of the same experiment, so existing
    # artifacts must stay in place. The original snapshot is preserved.
    if has_artifacts and bool(config.training.resume_from_latest_checkpoint):
        ensure_directories(config)
        if not Path(config.paths.path_to_config_snapshot).exists():
            save_config_snapshot(config)
        if not Path(config.paths.path_to_environment).exists():
            save_environment_info(config)
        return

    if has_artifacts:
        if not bool(config.general.overwrite_experiment):
            raise FileExistsError(
                f"Experiment '{config.general.experiment_name}' already exists at {experiment_dir}. "
                "Change experiment_name or set overwrite_experiment=True intentionally."
            )
        shutil.rmtree(experiment_dir)

    ensure_directories(config)
    save_config_snapshot(config)
    save_environment_info(config)


def file_sha256(path, chunk_size=1024 * 1024):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _array_info(array):
    if isinstance(array, dict):
        return {name: _array_info(value) for name, value in array.items()}
    array = np.asarray(array)
    return {
        "shape": [int(x) for x in array.shape],
        "dtype": str(array.dtype),
        "samples": int(len(array)),
    }


def _source_info(path, calculate_hash):
    p = Path(path)
    info = {"path": str(p), "size_bytes": p.stat().st_size if p.exists() else None, "sha256": None}
    if calculate_hash and p.exists():
        info["sha256"] = file_sha256(p)
    return info


def save_dataset_metadata(config, features, labels, groups=None, fold_ids=None):
    if not bool(config.reproducibility.save_data_info):
        return

    calculate_hash = bool(config.reproducibility.calculate_dataset_hash)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": str(config.general.experiment_name),
        "task": str(config.general.task),
        "features": _array_info(features),
        "labels": _array_info(labels),
        "multi_head": bool(config.strategies.multi_head.enabled),
        "metric_learning": bool(config.strategies.metric_learning.enabled),
        "finetuning": bool(config.strategies.finetuning.enabled),
        "self_training": bool(config.strategies.self_training.enabled),
        "groups": None if groups is None else _array_info(np.asarray(groups)),
        "fold_ids": None if fold_ids is None else _array_info(np.asarray(fold_ids)),
        "sources": {
            "features": _source_info(config.paths.path_to_train_features, calculate_hash),
            "labels": _source_info(config.paths.path_to_train_labels, calculate_hash),
            "groups": None if not config.paths.path_to_groups else _source_info(config.paths.path_to_groups, calculate_hash),
            "folds": None if not config.paths.path_to_folds else _source_info(config.paths.path_to_folds, calculate_hash),
        },
    }
    Path(config.paths.path_to_metadata).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_experiment_result(config, scores, elapsed_seconds):
    timestamp = datetime.now().isoformat(timespec="seconds")
    mean_score = float(np.mean(scores)) if scores else float("nan")
    std_score = float(np.std(scores)) if scores else float("nan")

    row = [
        timestamp,
        config.general.experiment_name,
        config.general.task,
        config.model.name,
        config.loss.name,
        config.optimizer.name,
        config.metric.name,
        mean_score,
        std_score,
        elapsed_seconds,
    ]

    if bool(config.logging.csv_file):
        path = Path(config.paths.path_to_results_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow([
                    "datetime", "experiment", "task", "model", "loss", "optimizer",
                    "metric", "cv_mean", "cv_std", "time_seconds",
                ])
            writer.writerow(row)

    if bool(config.logging.txt_file):
        path = Path(config.paths.path_to_results_txt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(
                f"{config.general.experiment_name} | {config.model.name} | {config.loss.name} | "
                f"{config.metric.name}: {mean_score:.6f} ± {std_score:.6f} | {elapsed_seconds} s\n"
            )
