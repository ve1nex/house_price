import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from checkpointing import load_checkpoint, save_checkpoint
from artifact_io import save_predictions, update_metadata
from data import get_data_loader, get_fold_loaders, labels_length
from finetuning import maybe_unfreeze_backbone, setup_finetuning
from ema import ModelEMA
from losses import get_loss
from metrics import get_metric, get_primary_raw_outputs, get_primary_targets, is_improvement, outputs_to_predictions
from models import count_parameters, freeze_batchnorms, get_model
from multi_head import get_head_specs, get_primary_head, get_primary_num_outputs, get_primary_task, is_multi_head
from notifier import TelegramNotifier
from optimizers import get_optimizer
from schedulers import get_scheduler, step_scheduler
from tracking import ExperimentTracker
from utils import resolve_device, save_experiment_result
from utils import set_seed
from visualization import save_validation_plot, save_training_curves, save_cv_scores


def _autocast_context(device, enabled):
    return torch.autocast(device_type=device.type, enabled=bool(enabled) and device.type == "cuda")


def _make_grad_scaler(device, enabled):
    return torch.amp.GradScaler("cuda", enabled=bool(enabled) and device.type == "cuda")


def _move_labels(labels, device):
    if isinstance(labels, dict):
        return {name: value.to(device, non_blocking=True) for name, value in labels.items()}
    return labels.to(device, non_blocking=True)


def _to_numpy_structure(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy_structure(item) for key, item in value.items()}
    raise TypeError(f"Unsupported output type: {type(value)}")


def _concat_structures(items):
    first = items[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(items, axis=0)
    if isinstance(first, dict):
        return {key: _concat_structures([item[key] for item in items]) for key in first}
    raise TypeError(f"Unsupported structure type: {type(first)}")


def _labels_to_numpy(labels):
    if isinstance(labels, dict):
        return {name: value.detach().cpu().numpy() for name, value in labels.items()}
    return labels.detach().cpu().numpy()


def _opt(config, section, name):
    return bool(config.optimization.enabled) and bool(config.optimization[section][name].enabled)


def _validate_optimization(config):
    if not bool(config.optimization.enabled):
        return
    for section, names in [("memory", ["gradient_checkpointing"]), ("model_compression", ["quantization", "pruning", "knowledge_distillation"])]:
        for name in names:
            if bool(config.optimization[section][name].enabled):
                raise NotImplementedError(f"optimization.{section}.{name} is a placeholder; no implementation is activated")


def train_one_epoch(config, model, train_loader, optimizer, loss_function, scaler, device, scheduler, ema=None):
    model.train()
    if bool(config.strategies.finetuning.enabled) and bool(config.strategies.finetuning.freeze_batchnorms):
        freeze_batchnorms(model)

    total_loss = 0.0
    part_sums = {}
    accumulation = int(config.optimization.memory.gradient_accumulation.steps) if _opt(config, "memory", "gradient_accumulation") else 1
    if accumulation < 1:
        raise ValueError("gradient_accumulation.steps must be >= 1")
    optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(train_loader, desc="Train", leave=False) if config.logging.prints else train_loader

    for step, batch in enumerate(iterator):
        features = batch["features"].to(device, non_blocking=True)
        labels = _move_labels(batch["labels"], device)

        with _autocast_context(device, _opt(config, "speed", "amp")):
            outputs = model(features)
            loss, parts = loss_function(outputs, labels)
            window_size = min(accumulation, len(train_loader) - (step // accumulation) * accumulation)
            backward_loss = loss / window_size

        scaler.scale(backward_loss).backward()
        total_loss += float(loss.detach().cpu())
        for key, value in parts.items():
            part_sums[key] = part_sums.get(key, 0.0) + float(value)

        should_step = ((step + 1) % accumulation == 0) or ((step + 1) == len(train_loader))
        if should_step:
            if _opt(config, "training_control", "gradient_clipping"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.optimization.training_control.gradient_clipping.max_norm))
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and str(config.scheduler.interval) == "step":
                step_scheduler(config, scheduler)

    batches = max(1, len(train_loader))
    return total_loss / batches, {key: value / batches for key, value in part_sums.items()}


def validate_one_epoch(config, model, val_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    part_sums = {}
    outputs_all, targets_all = [], []
    iterator = tqdm(val_loader, desc="Valid", leave=False) if config.logging.prints else val_loader

    with torch.no_grad():
        for batch in iterator:
            features = batch["features"].to(device, non_blocking=True)
            labels = _move_labels(batch["labels"], device)
            with _autocast_context(device, _opt(config, "speed", "amp")):
                outputs = model(features)
                loss, parts = loss_function(outputs, labels)
            total_loss += float(loss.detach().cpu())
            for key, value in parts.items():
                part_sums[key] = part_sums.get(key, 0.0) + float(value)
            outputs_all.append(_to_numpy_structure(outputs))
            targets_all.append(_labels_to_numpy(labels))

    outputs_all = _concat_structures(outputs_all)
    targets_all = _concat_structures(targets_all)
    metric = get_metric(config, targets_all, outputs_all)
    batches = max(1, len(val_loader))
    return (
        total_loss / batches,
        metric,
        outputs_all,
        targets_all,
        {key: value / batches for key, value in part_sums.items()},
    )


def _fold_dir(checkpoint_root, fold):
    return Path(checkpoint_root) / f"fold_{fold}"


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
    enable_tracking=True,
):
    fold_started = time.perf_counter()
    _validate_optimization(config)
    set_seed(int(config.general.seed) + int(fold), deterministic=bool(config.reproducibility.deterministic))
    device = resolve_device(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_root = Path(checkpoint_root or config.paths.path_to_fold_checkpoints)
    fold_dir = _fold_dir(checkpoint_root, fold)
    if save_artifacts:
        fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_idx, val_idx = get_fold_loaders(features, labels, groups, fold_ids, config, fold)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError(f"Fold {fold} has no train or validation batches; check split, debug size and dataloader_params.drop_last")

    model = get_model(config).to(device)
    setup_finetuning(model, config, map_location=device)
    ema = ModelEMA(model, decay=float(config.optimization.training_control.ema.decay), update_after_step=int(config.optimization.training_control.ema.update_after_step)) if _opt(config, "training_control", "ema") else None
    if _opt(config, "speed", "compile"):
        model.compile(backend=str(config.optimization.speed.compile.backend), mode=str(config.optimization.speed.compile.mode))
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    loss_function = get_loss(config)
    scaler = _make_grad_scaler(device, _opt(config, "speed", "amp"))

    total_params, trainable_params = count_parameters(model)
    if config.logging.prints:
        print(f"Device: {device} | Parameters: {total_params:,} ({trainable_params:,} trainable)")

    tracker = ExperimentTracker(config, fold, run_tag=run_tag) if enable_tracking else None
    if tracker is not None:
        tracker.start()

    current_epoch = 0
    epochs_since_improvement = 0
    best_metric = None
    best_outputs = None
    best_targets = None

    last_path = fold_dir / "last.pt"
    best_path = fold_dir / "best.pt"
    log_path = fold_dir / "log.txt"
    if save_artifacts and bool(config.logging.txt_file) and not bool(config.training.resume_from_latest_checkpoint):
        log_path.write_text(f"Experiment: {config.general.experiment_name} | Fold: {fold} | Stage: {run_tag or 'base'}\n", encoding="utf-8")

    if save_artifacts and bool(config.training.resume_from_latest_checkpoint) and last_path.exists():
        checkpoint = load_checkpoint(last_path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        current_epoch = int(checkpoint["epoch"])
        epochs_since_improvement = int(checkpoint["epochs_since_improvement"])
        best_metric = checkpoint["best_metric"]
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
        if config.logging.prints:
            print(f"Resumed fold {fold} from epoch {current_epoch}")

    history_path = fold_dir / "history.json"
    if save_artifacts and bool(config.training.resume_from_latest_checkpoint) and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = {"train_loss": [], "val_loss": [], "metric": [], "lr": []}

    for epoch in range(current_epoch, int(config.training.num_epochs)):
        if maybe_unfreeze_backbone(model, config, epoch) and config.logging.prints:
            print(f"Fine-tuning: backbone unfrozen at epoch {epoch + 1}")

        start = time.time()
        train_loss, train_parts = train_one_epoch(config, model, train_loader, optimizer, loss_function, scaler, device, scheduler, ema=ema)
        evaluated_model = ema.shadow if ema is not None else model
        val_loss, current_metric, outputs, targets, val_parts = validate_one_epoch(config, evaluated_model, val_loader, loss_function, device)

        if scheduler is not None and str(config.scheduler.interval) == "epoch":
            monitor = val_loss if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else current_metric
            step_scheduler(config, scheduler, monitor)

        lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["metric"].append(current_metric)
        history["lr"].append(lr)

        improved = is_improvement(config, current_metric, best_metric)
        if improved or not bool(config.training.save_best):
            best_metric = current_metric
            epochs_since_improvement = 0
            best_outputs = outputs
            best_targets = targets
            if save_artifacts and bool(config.training.save_best):
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch + 1, current_metric, best_metric, epochs_since_improvement, ema=ema, evaluation_model=evaluated_model)
        else:
            epochs_since_improvement += 1

        if save_artifacts and bool(config.training.save_last):
            save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch + 1, current_metric, best_metric, epochs_since_improvement, ema=ema)

        tracked = {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/metric": current_metric,
            "lr": lr,
            **{f"train/{key}": value for key, value in train_parts.items() if key != "loss/total"},
            **{f"val/{key}": value for key, value in val_parts.items() if key != "loss/total"},
        }
        if tracker is not None:
            tracker.log_epoch(epoch + 1, tracked)

        # Optuna pruning hook. During tuning this lets clearly bad trials stop early.
        if trial is not None:
            trial.report(current_metric, step=int(trial_step_offset) + epoch)
            if trial.should_prune():
                if tracker is not None:
                    tracker.close()
                import optuna
                raise optuna.TrialPruned()

        elapsed = int(time.time() - start)
        if config.logging.prints:
            print(
                f"Fold {fold} | Epoch {epoch + 1}/{config.training.num_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | metric={current_metric:.4f} | "
                f"best={best_metric:.4f} | lr={lr:.2e} | {elapsed}s"
            )
        if save_artifacts and bool(config.logging.txt_file):
            with log_path.open("a", encoding="utf-8") as file:
                file.write(
                    f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"metric={current_metric:.6f} best_metric={best_metric:.6f} lr={lr:.8g} time={elapsed}s\n"
                )

        if _opt(config, "training_control", "early_stopping") and epochs_since_improvement >= int(config.optimization.training_control.early_stopping.patience):
            if config.logging.prints:
                print("Early stopping.")
            break
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if tracker is not None:
        tracker.close()

    selected_path = best_path if bool(config.training.save_best) and best_path.exists() else last_path
    if save_artifacts and selected_path.exists():
        load_checkpoint(selected_path, model, map_location=device)
        _, best_metric, best_outputs, best_targets, _ = validate_one_epoch(config, model, val_loader, loss_function, device)
    elif best_outputs is None:
        evaluated_model = ema.shadow if ema is not None else model
        _, best_metric, best_outputs, best_targets, _ = validate_one_epoch(config, evaluated_model, val_loader, loss_function, device)

    if save_artifacts:
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "fold": int(fold),
        "score": float(best_metric),
        "outputs": best_outputs,
        "targets": best_targets,
        "val_idx": val_idx,
        "history": history,
        "training_time_seconds": time.perf_counter() - fold_started,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "best_checkpoint": (
            str(selected_path) if save_artifacts and selected_path.exists() else None
        ),
    }


def train_all_data(config, features, labels, checkpoint_root=None, run_tag=None):
    _validate_optimization(config)
    set_seed(int(config.general.seed) + 99991, deterministic=bool(config.reproducibility.deterministic))
    device = resolve_device(config)
    checkpoint_root = Path(checkpoint_root or config.paths.path_to_fold_checkpoints)
    fold_dir = checkpoint_root / "all_data"
    fold_dir.mkdir(parents=True, exist_ok=True)
    loader = get_data_loader(features, labels, config, is_train=True)
    if len(loader) == 0:
        raise ValueError("All-data training has no batches; check batch_size and drop_last")

    model = get_model(config).to(device)
    setup_finetuning(model, config, map_location=device)
    ema = ModelEMA(model, decay=float(config.optimization.training_control.ema.decay), update_after_step=int(config.optimization.training_control.ema.update_after_step)) if _opt(config, "training_control", "ema") else None
    if _opt(config, "speed", "compile"):
        model.compile(backend=str(config.optimization.speed.compile.backend), mode=str(config.optimization.speed.compile.mode))
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    loss_function = get_loss(config)
    scaler = _make_grad_scaler(device, _opt(config, "speed", "amp"))

    last_path = fold_dir / "last.pt"
    best_path = fold_dir / "best.pt"
    current_epoch = 0
    if bool(config.training.resume_from_latest_checkpoint) and last_path.exists():
        checkpoint = load_checkpoint(last_path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location=device)
        current_epoch = int(checkpoint["epoch"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])

    for epoch in range(current_epoch, int(config.training.num_epochs)):
        maybe_unfreeze_backbone(model, config, epoch)
        train_loss, _ = train_one_epoch(config, model, loader, optimizer, loss_function, scaler, device, scheduler, ema=ema)
        if scheduler is not None and str(config.scheduler.interval) == "epoch":
            step_scheduler(config, scheduler, train_loss)
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch + 1, None, None, 0, ema=ema)
        if config.logging.prints:
            print(f"All data | Epoch {epoch + 1}/{config.training.num_epochs} | train={train_loss:.4f}")

    save_checkpoint(best_path, model, optimizer, scheduler, scaler, int(config.training.num_epochs), None, None, 0, ema=ema, evaluation_model=ema.shadow if ema is not None else model)
    return [{"fold": "all_data", "score": float("nan"), "outputs": None, "targets": None, "val_idx": np.array([], dtype=int), "best_checkpoint": str(best_path)}]


def _save_oof(config, labels, fold_results, oof_path):
    if not fold_results:
        return
    n = labels_length(labels)
    first = fold_results[0]["outputs"]

    arrays = {}
    if isinstance(first, dict):
        # Save supervised outputs and optional embeddings separately.
        if "heads" in first:
            for head, values in first["heads"].items():
                shape = (n,) + tuple(values.shape[1:])
                arrays[f"outputs_{head}"] = np.full(shape, np.nan, dtype=np.float32)
            for head, values in labels.items():
                arrays[f"labels_{head}"] = np.asarray(values)
        else:
            values = first["logits"]
            arrays["outputs"] = np.full((n,) + tuple(values.shape[1:]), np.nan, dtype=np.float32)
            arrays["labels"] = np.asarray(labels)
        if "embeddings" in first:
            emb = first["embeddings"]
            arrays["embeddings"] = np.full((n,) + tuple(emb.shape[1:]), np.nan, dtype=np.float32)
    else:
        arrays["outputs"] = np.full((n,) + tuple(first.shape[1:]), np.nan, dtype=np.float32)
        arrays["labels"] = np.asarray(labels)

    for result in fold_results:
        idx = result["val_idx"]
        out = result["outputs"]
        if isinstance(out, dict):
            if "heads" in out:
                for head, values in out["heads"].items():
                    arrays[f"outputs_{head}"][idx] = values
            else:
                arrays["outputs"][idx] = out["logits"]
            if "embeddings" in out:
                arrays["embeddings"][idx] = out["embeddings"]
        else:
            arrays["outputs"][idx] = out

    Path(oof_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(oof_path, **arrays)


def train(
    config,
    features,
    labels,
    groups=None,
    fold_ids=None,
    checkpoint_root=None,
    oof_path=None,
    run_tag=None,
    save_global_result=True,
    send_notifications=True,
    ids=None,
    save_ensemble_artifacts=True,
):
    start_time = time.perf_counter()
    notifier = TelegramNotifier(config) if send_notifications else None
    checkpoint_root = checkpoint_root or config.paths.path_to_fold_checkpoints
    oof_path = oof_path or config.paths.path_to_oof_raw

    if bool(config.split.all_data_train):
        results = train_all_data(config, features, labels, checkpoint_root=checkpoint_root, run_tag=run_tag)
        elapsed = time.perf_counter() - start_time
        if save_global_result:
            update_metadata(config.paths.path_to_metadata, oof_complete=False, n_oof=0, n_train=labels_length(labels), training_time_seconds=elapsed)
        if notifier:
            notifier.send(f"{config.general.experiment_name} all-data training finished | {elapsed}s")
        return results

    folds = [int(fold) for fold in config.split.folds_to_train]
    scores, fold_results = [], []
    if config.logging.prints:
        print("Have a nice training!")
        print(f"Experiment: {config.general.experiment_name} | Stage: {run_tag or 'base'}")
        print(f"Folds to train: {folds}")

    for fold in folds:
        if config.logging.prints:
            print(f"\n--- Fold {fold} ---")
        result = run_fold(config, features, labels, groups, fold_ids, fold, checkpoint_root=checkpoint_root, run_tag=run_tag)
        scores.append(result["score"])
        fold_results.append(result)
        if notifier:
            notifier.send(f"{config.general.experiment_name} | {run_tag or 'base'} | Fold {fold} | score: {result['score']:.4f}")

    if save_ensemble_artifacts:
        n = labels_length(labels)
        ids = np.arange(n) if ids is None else np.asarray(ids)
        target = np.asarray(get_primary_targets(config, labels))
        task = get_primary_task(config)
        classes = list(range(get_primary_num_outputs(config))) if task == "classification" else None
        raw_predictions = np.full((n, len(classes)) if classes is not None else n, np.nan, dtype=float)
        folds_array = np.full(n, -1, dtype=int)
        for result in fold_results:
            idx = result["val_idx"]
            raw = np.asarray(get_primary_raw_outputs(config, result["outputs"]))
            if task == "classification":
                shifted = raw - raw.max(axis=1, keepdims=True)
                exponential = np.exp(shifted)
                values = exponential / exponential.sum(axis=1, keepdims=True)
            else:
                values = raw.reshape(-1)
            raw_predictions[idx] = values
            folds_array[idx] = int(result["fold"])
        selected = folds_array >= 0
        probabilities = raw_predictions[selected]
        artifact_values = probabilities[:, 1] if classes is not None and len(classes) == 2 else probabilities
        save_predictions(config.paths.path_to_oof, ids[selected], artifact_values, task, classes,
                         target[selected], folds_array[selected])
        _save_oof(config, labels, fold_results, oof_path)
        labels_pred = (np.argmax(probabilities, axis=1) if classes is not None else probabilities.reshape(-1))
        if bool(config.visualization.save_validation_plot):
            save_validation_plot(target[selected], labels_pred, task, config.paths.path_to_plots)
        if bool(config.visualization.save_cv_scores):
            save_cv_scores([(r["fold"], r["score"]) for r in fold_results], str(config.metric.name), config.paths.path_to_plots)
        if bool(config.visualization.save_training_curves):
            save_training_curves([r["history"] for r in fold_results], str(config.metric.name),
                                 Path(config.paths.path_to_plots) / "training_curves.png",
                                 show_lr=_opt(config, "training_control", "scheduler"))

    elapsed = time.perf_counter() - start_time
    if save_global_result:
        save_experiment_result(config, scores, elapsed)
        update_metadata(config.paths.path_to_metadata, task=get_primary_task(config),
                        classes=list(range(get_primary_num_outputs(config))) if get_primary_task(config) == "classification" else None,
                        id_namespace=str(config.data.id_namespace), metric=str(config.metric.name),
                        metric_direction=str(config.metric.direction),
                        fold_scores={str(r["fold"]):r["score"] for r in fold_results},
                        cv_mean=float(np.mean(scores)), cv_std=float(np.std(scores)),
                        oof_complete=bool(save_ensemble_artifacts and sum(len(r["val_idx"]) for r in fold_results) == labels_length(labels)),
                        n_oof=sum(len(r["val_idx"]) for r in fold_results), n_train=labels_length(labels),
                        training_time_seconds=elapsed,
                        fold_training_time_seconds={str(r["fold"]):r["training_time_seconds"] for r in fold_results},
                        peak_memory_mb=max((r["peak_memory_mb"] or 0 for r in fold_results), default=0) if torch.cuda.is_available() else None)

    cv_mean = float(np.mean(scores)) if scores else float("nan")
    cv_std = float(np.std(scores)) if scores else float("nan")
    if config.logging.prints:
        print(f"\nTraining finished. CV: {cv_mean:.4f} ± {cv_std:.4f} | {elapsed:.1f}s")
    if notifier:
        notifier.send(f"{config.general.experiment_name} | {run_tag or 'base'} finished | CV: {cv_mean:.4f} ± {cv_std:.4f} | {elapsed}s")
    return fold_results
