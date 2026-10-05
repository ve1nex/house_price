import torch
from omegaconf import OmegaConf


def get_optimizer(config, model):
    """Build the chosen optimizer with the configured weight decay."""
    name = str(config.optimizer.name)
    if not hasattr(torch.optim, name):
        raise ValueError(f"Unknown optimizer: {name}")
    params = OmegaConf.to_container(config.optimizer.params, resolve=True)
    params["weight_decay"] = (
        float(config.regularization.weight_decay.value)
        if config.regularization.enabled and config.regularization.weight_decay.enabled
        else 0.0
    )
    return getattr(torch.optim, name)(model.parameters(), **params)
