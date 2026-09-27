import numpy as np

from data import labels_length
from multi_head import get_head_specs, get_primary_head, is_multi_head


def check_training_data(features, labels, config):
    if int(config.training.num_epochs) < 1:
        raise ValueError("training.num_epochs must be >= 1")
    if int(config.dataloader_params.batch_size) < 1:
        raise ValueError("dataloader_params.batch_size must be >= 1")
    if not (bool(config.training.save_best) or bool(config.training.save_last)):
        raise ValueError("Enable training.save_best or training.save_last so inference has a checkpoint")
    if len(features) == 0:
        raise ValueError("Training features are empty.")
    if len(features) != labels_length(labels):
        raise ValueError("Features and labels have different lengths.")

    if is_multi_head(config):
        specs = get_head_specs(config)
        for name, spec in specs.items():
            if len(labels[name]) != len(features):
                raise ValueError(f"Head '{name}' has a different number of labels")
            if str(spec["task"]) == "classification" and len(np.unique(labels[name])) < 2:
                raise ValueError(f"Classification head '{name}' requires at least two classes")
    elif str(config.general.task) == "classification":
        if len(np.unique(labels)) < 2:
            raise ValueError("Classification requires at least two classes.")
    primary = labels[get_primary_head(config)] if is_multi_head(config) else labels
    primary_task = str(get_head_specs(config)[get_primary_head(config)]["task"]) if is_multi_head(config) else str(config.general.task)
    class_count = int(get_head_specs(config)[get_primary_head(config)]["num_outputs"]) if is_multi_head(config) else int(config.general.num_classes)
    if primary_task == "classification" and not np.array_equal(np.unique(primary), np.arange(class_count)):
        raise ValueError("Classification labels must be integer-encoded 0..num_classes-1; encode before training")

    if bool(config.strategies.metric_learning.enabled):
        if is_multi_head(config):
            label_head = config.strategies.metric_learning.label_head
            label_head = str(label_head) if label_head else get_primary_head(config)
            task = str(get_head_specs(config)[label_head]["task"])
        else:
            task = str(config.general.task)
        if task != "classification":
            raise ValueError("The included triplet metric-learning implementation requires class labels")
    if bool(config.strategies.hard_negative_mining.enabled) and not bool(config.strategies.metric_learning.enabled):
        raise ValueError("hard_negative_mining requires strategies.metric_learning.enabled=True")

    if bool(config.strategies.self_training.enabled):
        if is_multi_head(config):
            raise ValueError("Base self-training implementation currently supports single-head classification only")
        if str(config.general.task) != "classification":
            raise ValueError("Base self-training implementation currently supports classification only")
        if str(config.split.strategy) in {"GroupKFold", "StratifiedGroupKFold"}:
            raise ValueError("Base self-training implementation does not invent groups for unlabeled data; use non-group CV or customize self_training.py")
