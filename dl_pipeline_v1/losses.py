import torch.nn as nn
from omegaconf import OmegaConf


def get_loss(config):
    """Instantiate a regression loss; outputs and targets are flattened by the loop."""
    name = str(config.loss.name)
    if not hasattr(nn, name):
        raise ValueError(f"Unknown loss: {name}")
    return getattr(nn, name)(**OmegaConf.to_container(config.loss.params, resolve=True))
