import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from artifact_io import save_predictions, update_metadata
from checkpointing import load_checkpoint, save_checkpoint
from data import get_fold_loaders
from losses import get_loss
from metrics import get_metric, is_improvement
from models import count_parameters, get_model
from optimizers import get_optimizer
from schedulers import get_scheduler, step_scheduler
from utils import resolve_device, save_experiment_result, set_seed
from visualization import save_cv_scores, save_training_curves, save_validation_plot


def _opt(config, name):
    """Apply the training-control master switch to one feature flag."""
    return bool(
        config.optimization.enabled
        and config.optimization.training_control[name].enabled
    )


def train_one_epoch(config, model, loader, optimizer, loss_function, device, scheduler):
    """Perform ordinary MLP updates with optional gradient clipping."""
    model.train()
    total, samples = 0.0, 0
    iterator = (
        tqdm(loader, desc="Train", leave=False) if config.logging.prints else loader
    )
    for batch in iterator:
        x, y = batch["features"].to(device), batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x).reshape(-1), y.reshape(-1))
        loss.backward()
        if _opt(config, "gradient_clipping"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config.optimization.training_control.gradient_clipping.max_norm),
            )
        optimizer.step()
        if scheduler is not None and str(config.scheduler.interval) == "step":
            step_scheduler(config, scheduler)
        total += float(loss.detach().cpu()) * len(y)
        samples += len(y)
    return total / samples


def validate_one_epoch(config, model, loader, loss_function, device):
    """Collect validation predictions without parameter updates."""
    model.eval()
    total, samples = 0.0, 0
    outputs, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["features"].to(device), batch["labels"].to(device)
            out = model(x).reshape(-1)
            total += float(loss_function(out, y.reshape(-1)).detach().cpu()) * len(y)
            samples += len(y)
            outputs.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    outputs, targets = np.concatenate(outputs), np.concatenate(targets)
    return total / samples, get_metric(config, targets, outputs), outputs, targets


def run_fold(
    config,
    features,
    labels,
    groups,
    fold_ids,
    fold,
    checkpoint_root=None,
    run_tag=None,
    trial=None,
    trial_step_offset=0,
    save_artifacts=True,
):
    """Train one fold with a preprocessor fitted exclusively on its training rows."""
    started = time.perf_counter()
    set_seed(
        int(config.general.seed) + int(fold), bool(config.reproducibility.deterministic)
    )
    device = resolve_device(config)
    directory = (
        Path(checkpoint_root or config.paths.path_to_fold_checkpoints) / f"fold_{fold}"
    )
    if save_artifacts:
        directory.mkdir(parents=True, exist_ok=True)
    train_loader, valid_loader, _, val_idx = get_fold_loaders(
        features,
        labels,
        groups,
        fold_ids,
        config,
        fold,
        preprocessor_path=directory / "preprocessor.joblib" if save_artifacts else None,
    )
    model = get_model(config).to(device)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    loss_function = get_loss(config)
    if config.logging.prints:
        print(
            f"Fold {fold} | device={device} | parameters={count_parameters(model)[0]:,}"
        )
    best_metric, best_outputs, best_targets = None, None, None
    patience = 0
    history = {"train_loss": [], "val_loss": [], "metric": [], "lr": []}
    best_path = directory / "best.pt"
    for epoch in range(int(config.training.num_epochs)):
        train_loss = train_one_epoch(
            config, model, train_loader, optimizer, loss_function, device, scheduler
        )
        val_loss, metric, outputs, targets = validate_one_epoch(
            config, model, valid_loader, loss_function, device
        )
        if scheduler is not None and str(config.scheduler.interval) == "epoch":
            step_scheduler(config, scheduler, val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        for name, value in [
            ("train_loss", train_loss),
            ("val_loss", val_loss),
            ("metric", metric),
            ("lr", lr),
        ]:
            history[name].append(value)
        if is_improvement(config, metric, best_metric):
            best_metric, best_outputs, best_targets, patience = (
                metric,
                outputs,
                targets,
                0,
            )
            if save_artifacts:
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    metric,
                    config.model.input_shape,
                )
        else:
            patience += 1
        if save_artifacts and config.training.save_last:
            save_checkpoint(
                directory / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                metric,
                config.model.input_shape,
            )
        if trial is not None:
            trial.report(metric, step=int(trial_step_offset) + epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
        if config.logging.prints:
            print(
                f"Fold {fold} | epoch={epoch + 1} | train={train_loss:.5f} | val={val_loss:.5f} | metric={metric:.5f}"
            )
        if _opt(config, "early_stopping") and patience >= int(
            config.optimization.training_control.early_stopping.patience
        ):
            break
    if save_artifacts:
        load_checkpoint(best_path, model, map_location=device)
        _, best_metric, best_outputs, best_targets = validate_one_epoch(
            config, model, valid_loader, loss_function, device
        )
        (directory / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if config.logging.txt_file:
            lines = [
                f"epoch={i + 1} train_loss={tr:.6f} val_loss={va:.6f} metric={m:.6f} lr={lr:.8g}"
                for i, (tr, va, m, lr) in enumerate(zip(*history.values()))
            ]
            (directory / "log.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
    return {
        "fold": int(fold),
        "score": float(best_metric),
        "outputs": best_outputs,
        "targets": best_targets,
        "val_idx": val_idx,
        "history": history,
        "training_time_seconds": time.perf_counter() - started,
        "best_checkpoint": str(best_path) if save_artifacts else None,
    }


def train(config, features, labels, groups=None, fold_ids=None, ids=None):
    """Save the fold models, raw OOF values, metric summaries, and task-aware plots."""
    started = time.perf_counter()
    folds = [int(f) for f in config.split.folds_to_train]
    if not folds or len(folds) != len(set(folds)):
        raise ValueError("Select unique folds")
    results = [run_fold(config, features, labels, groups, fold_ids, f) for f in folds]
    predictions = np.full(len(labels), np.nan)
    assignments = np.full(len(labels), -1, dtype=int)
    for result in results:
        predictions[result["val_idx"]] = result["outputs"]
        assignments[result["val_idx"]] = result["fold"]
    selected = assignments >= 0
    ids = np.arange(len(labels)) if ids is None else np.asarray(ids)
    save_predictions(
        config.paths.path_to_oof,
        ids[selected],
        predictions[selected],
        "regression",
        target=labels[selected],
        folds=assignments[selected],
    )
    np.savez(
        config.paths.path_to_oof_raw,
        outputs=predictions,
        labels=labels,
        row_index=np.arange(len(labels)),
        fold=assignments,
    )
    scores = [r["score"] for r in results]
    elapsed = time.perf_counter() - started
    save_experiment_result(config, scores, elapsed)
    update_metadata(
        config.paths.path_to_metadata,
        task="regression",
        classes=None,
        id_namespace=str(config.data.id_namespace),
        metric=str(config.metric.name),
        metric_direction=str(config.metric.direction),
        fold_scores={str(r["fold"]): r["score"] for r in results},
        cv_mean=float(np.mean(scores)),
        cv_std=float(np.std(scores)),
        oof_complete=bool(selected.all()),
        n_oof=int(selected.sum()),
        n_train=len(labels),
        preprocessing="per_fold",
        training_time_seconds=elapsed,
    )
    if config.visualization.save_validation_plot:
        save_validation_plot(
            labels[selected],
            predictions[selected],
            "regression",
            config.paths.path_to_plots,
        )
    if config.visualization.save_cv_scores:
        save_cv_scores(
            [(r["fold"], r["score"]) for r in results],
            str(config.metric.name),
            config.paths.path_to_plots,
        )
    if config.visualization.save_training_curves:
        save_training_curves(
            [r["history"] for r in results],
            str(config.metric.name),
            Path(config.paths.path_to_plots) / "training_curves.png",
            show_lr=_opt(config, "scheduler"),
        )
    if config.logging.prints:
        print(f"CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return results
