from pathlib import Path

from omegaconf import OmegaConf


class ExperimentTracker:
    """Optional W&B/TensorBoard mirror. Local files remain the source of truth."""

    def __init__(self, config, fold, run_tag=None):
        self.config = config
        self.fold = fold
        self.run_tag = run_tag
        self.wandb_run = None
        self.writer = None

    @property
    def run_name(self):
        suffix = f"{self.run_tag}_" if self.run_tag else ""
        return f"{self.config.general.experiment_name}_{suffix}fold_{self.fold}"

    def start(self):
        if bool(self.config.tracking.wandb):
            try:
                import wandb
            except ImportError as error:
                raise ImportError("W&B logging is enabled. Install 'wandb'.") from error
            kwargs = {
                "project": str(self.config.tracking.wandb_project_name),
                "entity": self.config.tracking.wandb_username,
                "name": self.run_name,
                "reinit": True,
            }
            if bool(self.config.tracking.wandb_log_config):
                kwargs["config"] = OmegaConf.to_container(self.config, resolve=True)
            self.wandb_run = wandb.init(**kwargs)

        if bool(self.config.tracking.tensorboard):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise ImportError("TensorBoard logging is enabled. Install 'tensorboard'.") from error
            suffix = Path(self.run_tag) if self.run_tag else Path()
            log_dir = Path(self.config.paths.tensorboard_dir) / suffix / f"fold_{self.fold}"
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_epoch(self, epoch, values):
        clean = {key: float(value) for key, value in values.items()}
        if self.wandb_run is not None:
            self.wandb_run.log({**clean, "epoch": epoch})
        if self.writer is not None:
            for key, value in clean.items():
                self.writer.add_scalar(key, value, epoch)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
