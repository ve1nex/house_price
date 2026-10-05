import numpy as np


def check_training_data(features, labels, config):
    """Reject invalid training sizes, folds, and non-finite regression targets."""
    if len(features) != len(labels) or not len(labels):
        raise ValueError(
            "Training features and labels must be non-empty and have equal length"
        )
    if not np.isfinite(labels).all():
        raise ValueError("Targets contain NaN or infinity")
    if (
        int(config.training.num_epochs) < 1
        or int(config.dataloader_params.batch_size) < 1
    ):
        raise ValueError("num_epochs and batch_size must be positive")
    if not config.training.save_best:
        raise ValueError(
            "save_best is required to match OOF and test checkpoint selection"
        )
    if not config.split.folds_to_train:
        raise ValueError("Select at least one fold")
