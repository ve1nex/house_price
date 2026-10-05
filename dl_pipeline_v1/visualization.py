"""House Prices training and OOF diagnostics."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _mean_curve(histories, name):
    """Average epoch values over folds that reached each epoch."""
    length = max((len(h[name]) for h in histories))
    padded = np.full((len(histories), length), np.nan)
    for index, history in enumerate(histories):
        padded[index, : len(history[name])] = history[name]
    return np.nanmean(padded, axis=0)


def save_training_curves(histories, metric_name, output_path, show_lr=False):
    """Plot saved training loss, validation loss, metric, and learning rate."""
    if not histories or not any((h["train_loss"] for h in histories)):
        return
    panels = 3 if show_lr else 2
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4))
    for key, label in [("train_loss", "Train"), ("val_loss", "Validation")]:
        curve = _mean_curve(histories, key)
        axes[0].plot(np.arange(1, len(curve) + 1), curve, label=label)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training / validation loss")
    axes[0].legend()
    metric = _mean_curve(histories, "metric")
    axes[1].plot(np.arange(1, len(metric) + 1), metric)
    axes[1].set(xlabel="Epoch", ylabel=metric_name, title="Validation metric")
    if show_lr:
        curve = _mean_curve(histories, "lr")
        axes[2].plot(np.arange(1, len(curve) + 1), curve)
        axes[2].set(xlabel="Epoch", ylabel="Learning rate", title="LR")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_validation_plot(y_true, y_pred, task, output_dir):
    """Save validation diagnostics for the current prediction task."""
    if len(y_true) == 0:
        return
    y_true, y_pred = (np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(y_true, y_pred, alpha=0.55)
    low, high = (min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()))
    ax1.plot([low, high], [low, high], "--", color="black")
    ax1.set(xlabel="True", ylabel="Predicted", title="OOF true vs predicted")
    ax2.scatter(y_pred, y_true - y_pred, alpha=0.55)
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set(xlabel="Predicted", ylabel="True − predicted", title="Residuals")
    fig.tight_layout()
    path = Path(output_dir) / "validation.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_cv_scores(fold_scores, metric_name, output_dir):
    """Plot validation fold scores with their mean and standard deviation."""
    if not fold_scores:
        return
    labels, scores = zip(*fold_scores)
    scores = np.asarray(scores, dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(scores)), 5))
    bars = ax.bar([str(f) for f in labels], scores)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.axhline(
        scores.mean(),
        color="darkred",
        linestyle="--",
        label=f"Mean ± std: {scores.mean():.4f} ± {scores.std():.4f}",
    )
    ax.set(xlabel="Fold", ylabel=metric_name, title="CV scores")
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "cv_scores.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
