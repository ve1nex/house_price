import torch
from omegaconf import OmegaConf


def get_scheduler(config, optimizer):
    """Instantiate the scheduler when its feature flag is enabled."""
    if not (
        config.optimization.enabled
        and config.optimization.training_control.scheduler.enabled
    ):
        return None
    name = str(config.scheduler.name)
    if not hasattr(torch.optim.lr_scheduler, name):
        raise ValueError(f"Unknown scheduler: {name}")
    return getattr(torch.optim.lr_scheduler, name)(
        optimizer, **OmegaConf.to_container(config.scheduler.params, resolve=True)
    )


def step_scheduler(config, scheduler, metric=None):
    """Pass validation loss to plateau schedulers and advance other schedulers normally."""
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    elif scheduler is not None:
        scheduler.step()
