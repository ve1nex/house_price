from pathlib import Path

from omegaconf import OmegaConf


_ACTIVE_RUN = None


def _enabled(config) -> bool:
    return bool(config.tracking.wandb) and str(config.tracking.wandb_mode) != "disabled"


def _handle_error(config, message: str, error: Exception):
    if bool(config.tracking.raise_on_error):
        raise error
    print(f"W&B tracking warning: {message}: {error}")


def start_tracking(config):
    """Start one W&B run for the current experiment.

    Local checkpoints/logs remain independent from W&B. If W&B is disabled or
    unavailable, the training pipeline can continue normally.
    """
    global _ACTIVE_RUN

    if not _enabled(config):
        return None

    try:
        import wandb

        Path(config.paths.path_to_wandb).mkdir(parents=True, exist_ok=True)
        resolved_config = OmegaConf.to_container(config, resolve=True)

        entity = config.tracking.wandb_entity
        if entity in (None, "", "null"):
            entity = None

        _ACTIVE_RUN = wandb.init(
            project=str(config.tracking.wandb_project),
            entity=entity,
            name=str(config.general.experiment_name),
            config=resolved_config,
            tags=list(config.tracking.wandb_tags),
            dir=str(config.paths.path_to_wandb),
            mode=str(config.tracking.wandb_mode),
        )
        return _ACTIVE_RUN
    except Exception as error:
        _handle_error(config, "could not start run", error)
        return None


def log_metrics(config, metrics: dict, step: int | None = None) -> None:
    """Log scalar metrics to the active W&B run."""
    if _ACTIVE_RUN is None:
        return
    try:
        _ACTIVE_RUN.log(metrics, step=step)
    except Exception as error:
        _handle_error(config, "could not log metrics", error)


def log_summary(config, values: dict) -> None:
    """Store final scalar values in the W&B run summary."""
    if _ACTIVE_RUN is None:
        return
    try:
        for key, value in values.items():
            _ACTIVE_RUN.summary[key] = value
    except Exception as error:
        _handle_error(config, "could not write summary", error)


def log_experiment_artifacts(config) -> None:
    """Upload the locally saved experiment folder as a W&B artifact."""
    if _ACTIVE_RUN is None or not bool(config.tracking.log_artifacts):
        return

    try:
        import wandb

        experiment_dir = Path(config.paths.path_to_checkpoints)
        if not experiment_dir.exists():
            return

        artifact = wandb.Artifact(
            name=f"{config.general.experiment_name}-artifacts",
            type="classic-ml-experiment",
            metadata={
                "task": str(config.general.task),
                "metric": str(config.metric.name),
            },
        )
        artifact.add_dir(str(experiment_dir))
        _ACTIVE_RUN.log_artifact(artifact)

        if bool(config.tracking.log_plots):
            plots_dir = Path(config.paths.path_to_plots)
            for plot_path in sorted(plots_dir.glob("*.png")):
                _ACTIVE_RUN.log({f"plots/{plot_path.stem}": wandb.Image(str(plot_path))})
    except Exception as error:
        _handle_error(config, "could not upload artifacts", error)


def finish_tracking(config, exit_code: int = 0) -> None:
    """Finish the active W&B run if one exists."""
    global _ACTIVE_RUN
    if _ACTIVE_RUN is None:
        return
    try:
        _ACTIVE_RUN.finish(exit_code=exit_code)
    except Exception as error:
        _handle_error(config, "could not finish run", error)
    finally:
        _ACTIVE_RUN = None
