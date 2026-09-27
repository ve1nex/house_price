import torch
from omegaconf import OmegaConf


def get_scheduler(config, optimizer):
    """Create a standard PyTorch scheduler with optional standard warmup."""
    if not (bool(config.optimization.enabled) and bool(config.optimization.training_control.scheduler.enabled)):
        if bool(config.optimization.enabled) and bool(config.optimization.training_control.warmup.enabled):
            raise ValueError("Warmup requires optimization.training_control.scheduler.enabled=True")
        return None

    name = str(config.scheduler.name)
    if not hasattr(torch.optim.lr_scheduler, name):
        raise ValueError(f"Unknown torch scheduler: {name}")

    params = OmegaConf.to_container(config.scheduler.params, resolve=True)
    main_scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **params)

    warmup = config.scheduler.warmup
    if not bool(config.optimization.training_control.warmup.enabled):
        return main_scheduler

    if str(config.scheduler.interval) != "epoch":
        raise ValueError("Base warmup implementation currently expects scheduler.interval='epoch'.")

    warmup_epochs = int(warmup.epochs)
    if isinstance(main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        raise ValueError("ReduceLROnPlateau cannot be combined with this SequentialLR warmup")
    if warmup_epochs <= 0:
        return main_scheduler

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(warmup.start_factor),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )


def step_scheduler(config, scheduler, metric=None):
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric is None:
            raise ValueError("ReduceLROnPlateau requires a validation metric/loss.")
        scheduler.step(metric)
    else:
        scheduler.step()
