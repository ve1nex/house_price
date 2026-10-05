from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, metric, input_shape):
    """Store weights, optimizer state, and the fold-specific transformed input shape."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": int(epoch),
            "metric": float(metric),
            "input_shape": list(input_shape),
            "preprocessing": "per_fold",
        },
        path,
    )


def load_checkpoint(path, model, map_location="cpu"):
    """Load both the original checkpoints and new fold-local checkpoints."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return checkpoint
