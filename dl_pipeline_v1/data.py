from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from prepare_house_prices import load_raw_data, prepare_fold


def _load_array(path):
    """Load one of the original numeric arrays used by archived MLP checkpoints."""
    return np.load(Path(path), allow_pickle=False)


def load_training_data(config):
    """Load unscaled raw training features so validation statistics remain unseen."""
    features, labels, _ = load_raw_data(config, train=True)
    folds = np.full(len(labels), -1, dtype=int)
    cv = KFold(
        n_splits=int(config.split.n_splits),
        shuffle=bool(config.split.shuffle),
        random_state=int(config.general.seed) if config.split.shuffle else None,
    )
    for fold, (_, va) in enumerate(cv.split(features)):
        folds[va] = fold
    return features, labels, None, folds


def load_test_data(config):
    """Return unscaled test features for the per-fold preprocessing pipelines."""
    return load_raw_data(config, train=False)[0]


class TensorDataset(Dataset):
    """Expose dense tabular samples and scalar regression targets to PyTorch."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {
            "features": torch.as_tensor(self.features[index], dtype=torch.float32),
            "labels": torch.as_tensor(self.labels[index], dtype=torch.float32),
        }


def get_data_loader(features, labels, config, is_train, seed=None):
    """Create a reproducibly shuffled loader for already transformed samples."""
    generator = torch.Generator().manual_seed(
        int(config.general.seed) if seed is None else int(seed)
    )
    return DataLoader(
        TensorDataset(features, labels),
        generator=generator,
        batch_size=int(config.dataloader_params.batch_size),
        shuffle=bool(config.dataloader_params.shuffle) if is_train else False,
        drop_last=bool(config.dataloader_params.drop_last) if is_train else False,
    )


def get_inference_loader(features, config):
    """Create an ordered test loader."""
    return DataLoader(
        torch.as_tensor(features, dtype=torch.float32),
        batch_size=int(config.dataloader_params.batch_size),
        shuffle=False,
    )


def get_fold_indices(features, labels, groups, fold_ids, config, current_fold):
    """Select a fold from the shared House Prices KFold assignment."""
    if fold_ids is None:
        cv = KFold(
            n_splits=int(config.split.n_splits),
            shuffle=bool(config.split.shuffle),
            random_state=int(config.general.seed) if config.split.shuffle else None,
        )
        splits = list(cv.split(features))
        if not 0 <= int(current_fold) < len(splits):
            raise ValueError(f"Invalid fold: {current_fold}")
        return splits[int(current_fold)]
    train_idx, val_idx = (
        np.flatnonzero(fold_ids != current_fold),
        np.flatnonzero(fold_ids == current_fold),
    )
    if not len(train_idx) or not len(val_idx):
        raise ValueError(f"Fold {current_fold} has an empty split")
    return train_idx, val_idx


def get_fold_loaders(
    features, labels, groups, fold_ids, config, current_fold, preprocessor_path=None
):
    """Fit fold-local preprocessing and persist it beside the matching checkpoint."""
    tr, va = get_fold_indices(features, labels, groups, fold_ids, config, current_fold)
    x_train, x_valid, preprocessor = prepare_fold(features, tr, va)
    config.model.input_shape = [int(x_train.shape[1])]
    if preprocessor_path is not None:
        Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
    seed = int(config.general.seed) + int(current_fold) * 10007
    return (
        get_data_loader(x_train, labels[tr], config, True, seed),
        get_data_loader(x_valid, labels[va], config, False, seed + 1),
        tr,
        va,
    )
