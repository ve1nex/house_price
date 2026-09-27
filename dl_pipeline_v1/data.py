from pathlib import Path
import random

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from augmentations import apply_augmentations
from multi_head import get_head_specs, get_primary_head, get_primary_task, is_multi_head
from preprocessing import preprocess_sample
from samplers import get_balanced_batch_sampler


def _load_array(path, key=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if key is not None:
            return archive[key]
        if len(archive.files) != 1:
            return archive[archive.files[0]]
        return archive[archive.files[0]]
    if suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, dict) and key is not None:
            value = value[key]
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    raise ValueError(f"Unsupported data format: {suffix}. Use .npy, .npz or .pt/.pth")


def _load_labels(config):
    path = Path(config.paths.path_to_train_labels)
    if not is_multi_head(config):
        return np.asarray(_load_array(path, config.data.label_key))

    if path.suffix.lower() != ".npz":
        raise ValueError("multi_head.enabled=True requires path_to_train_labels to be an .npz archive")
    archive = np.load(path, allow_pickle=False)
    specs = get_head_specs(config)
    missing = [name for name in specs if name not in archive.files]
    if missing:
        raise ValueError(f"Multi-head label archive is missing arrays: {missing}")
    return {name: np.asarray(archive[name]) for name in specs}


def load_training_data(config):
    features = np.asarray(_load_array(config.paths.path_to_train_features, config.data.feature_key))
    labels = _load_labels(config)

    if len(features) != labels_length(labels):
        raise ValueError("Features and labels must contain the same number of samples.")

    groups = None
    if config.paths.path_to_groups:
        groups = np.asarray(_load_array(config.paths.path_to_groups))
        if len(groups) != labels_length(labels):
            raise ValueError("Groups and labels must contain the same number of samples.")

    fold_ids = None
    if bool(config.split.already_split):
        if not config.paths.path_to_folds:
            raise ValueError("split.already_split=True requires paths.path_to_folds")
        fold_ids = np.asarray(_load_array(config.paths.path_to_folds))
        if len(fold_ids) != labels_length(labels):
            raise ValueError("Fold ids and labels must contain the same number of samples.")

    return features, labels, groups, fold_ids


def load_test_data(config):
    return np.asarray(_load_array(config.paths.path_to_test_features, config.data.feature_key))


def load_unlabeled_data(config):
    path = config.paths.path_to_unlabeled_features
    if not path:
        raise ValueError("self_training.enabled=True requires paths.path_to_unlabeled_features")
    return np.asarray(_load_array(path, config.data.feature_key))


def labels_length(labels):
    if isinstance(labels, dict):
        first = next(iter(labels.values()))
        return len(first)
    return len(labels)


def slice_labels(labels, indices):
    if isinstance(labels, dict):
        return {name: np.asarray(value)[indices] for name, value in labels.items()}
    return np.asarray(labels)[indices]


def get_primary_labels(labels, config):
    if isinstance(labels, dict):
        return np.asarray(labels[get_primary_head(config)])
    return np.asarray(labels)


def _label_to_tensor(value, task):
    if task == "classification":
        return torch.as_tensor(value, dtype=torch.long)
    return torch.as_tensor(value, dtype=torch.float32)


class TensorDataset(Dataset):
    def __init__(self, features, labels, config, is_train: bool):
        self.features = features
        self.labels = labels
        self.config = config
        self.is_train = is_train

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.as_tensor(self.features[index], dtype=torch.float32)
        x = preprocess_sample(x, self.config, is_train=self.is_train)
        if self.is_train:
            x = apply_augmentations(x, self.config)

        if isinstance(self.labels, dict):
            specs = get_head_specs(self.config)
            y = {
                name: _label_to_tensor(self.labels[name][index], str(specs[name]["task"]))
                for name in specs
            }
        else:
            y = _label_to_tensor(self.labels[index], str(self.config.general.task))
        return {"features": x, "labels": y}


class InferenceDataset(Dataset):
    def __init__(self, features, config):
        self.features = features
        self.config = config

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.as_tensor(self.features[index], dtype=torch.float32)
        return preprocess_sample(x, self.config, is_train=False)


def _loader_params(config, is_train: bool):
    optimized = bool(config.optimization.enabled) and bool(config.optimization.speed.optimized_dataloader.enabled)
    advanced = config.optimization.speed.optimized_dataloader
    workers = int(advanced.num_workers) if optimized else 0
    params = {
        "batch_size": int(config.dataloader_params.batch_size),
        "num_workers": workers,
        "pin_memory": bool(advanced.pin_memory) if optimized and torch.cuda.is_available() else False,
        "drop_last": bool(config.dataloader_params.drop_last) if is_train else False,
        "shuffle": bool(config.dataloader_params.shuffle) if is_train else False,
        "persistent_workers": bool(advanced.persistent_workers) if workers > 0 else False,
    }
    if workers > 0:
        params["prefetch_factor"] = int(advanced.prefetch_factor)
        params["worker_init_fn"] = _seed_worker
    return params


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def get_data_loader(features, labels, config, is_train: bool, seed=None):
    dataset = TensorDataset(features, labels, config, is_train=is_train)
    generator = torch.Generator().manual_seed(int(config.general.seed) if seed is None else int(seed))

    # Metric learning benefits from class-balanced batches. This replaces only
    # the batch construction; Dataset/preprocessing/train loop stay unchanged.
    if is_train and bool(config.strategies.metric_learning.enabled) and bool(config.strategies.metric_learning.balanced_batches):
        metric_labels = get_primary_labels(labels, config)
        sampler = get_balanced_batch_sampler(metric_labels, config, seed=seed)
        params = _loader_params(config, is_train=True)
        for key in ("batch_size", "drop_last", "shuffle"):
            params.pop(key)
        return DataLoader(dataset, batch_sampler=sampler, generator=generator, **params)

    return DataLoader(dataset, generator=generator, **_loader_params(config, is_train=is_train))


def get_inference_loader(features, config):
    dataset = InferenceDataset(features, config)
    generator = torch.Generator().manual_seed(int(config.general.seed))
    return DataLoader(dataset, generator=generator, **_loader_params(config, is_train=False))


def _get_splitter(config):
    strategy = str(config.split.strategy)
    common = {"n_splits": int(config.split.n_splits)}
    if strategy == "KFold":
        return KFold(**common, shuffle=bool(config.split.shuffle), random_state=int(config.general.seed) if config.split.shuffle else None)
    if strategy == "StratifiedKFold":
        return StratifiedKFold(**common, shuffle=bool(config.split.shuffle), random_state=int(config.general.seed) if config.split.shuffle else None)
    if strategy == "GroupKFold":
        return GroupKFold(**common)
    if strategy == "StratifiedGroupKFold":
        return StratifiedGroupKFold(**common, shuffle=bool(config.split.shuffle), random_state=int(config.general.seed) if config.split.shuffle else None)
    raise ValueError(f"Unknown split strategy: {strategy}")


def get_fold_indices(features, labels, groups, fold_ids, config, current_fold):
    if fold_ids is not None:
        val_idx = np.where(fold_ids == current_fold)[0]
        train_idx = np.where(fold_ids != current_fold)[0]
        return train_idx, val_idx

    splitter = _get_splitter(config)
    strategy = str(config.split.strategy)
    primary_labels = get_primary_labels(labels, config)

    if strategy == "KFold":
        iterator = splitter.split(features)
    elif strategy == "StratifiedKFold":
        iterator = splitter.split(features, primary_labels)
    elif strategy == "GroupKFold":
        if groups is None:
            raise ValueError("GroupKFold requires paths.path_to_groups")
        iterator = splitter.split(features, primary_labels, groups)
    elif strategy == "StratifiedGroupKFold":
        if groups is None:
            raise ValueError("StratifiedGroupKFold requires paths.path_to_groups")
        iterator = splitter.split(features, primary_labels, groups)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    for fold, (train_idx, val_idx) in enumerate(iterator):
        if fold == current_fold:
            return train_idx, val_idx
    raise ValueError(f"Fold {current_fold} does not exist")


def get_fold_loaders(features, labels, groups, fold_ids, config, current_fold):
    train_idx, val_idx = get_fold_indices(features, labels, groups, fold_ids, config, current_fold)

    if bool(config.training.debug):
        rng = np.random.default_rng(int(config.general.seed) + int(current_fold))
        train_n = min(int(config.training.number_of_train_debug_samples), len(train_idx))
        val_n = min(int(config.training.number_of_val_debug_samples), len(val_idx))
        train_idx = rng.choice(train_idx, size=train_n, replace=False)
        val_idx = rng.choice(val_idx, size=val_n, replace=False)

    base_seed = int(config.general.seed) + int(current_fold) * 10007
    train_loader = get_data_loader(features[train_idx], slice_labels(labels, train_idx), config, is_train=True, seed=base_seed)
    val_loader = get_data_loader(features[val_idx], slice_labels(labels, val_idx), config, is_train=False, seed=base_seed + 1)
    return train_loader, val_loader, np.asarray(train_idx), np.asarray(val_idx)
