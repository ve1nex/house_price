import json
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from checkpointing import load_checkpoint
from data import _load_array, get_inference_loader, load_test_data
from artifact_io import save_predictions, update_metadata
from models import get_model
from multi_head import get_head_specs, get_primary_head, get_primary_num_outputs, get_primary_task, is_multi_head
from postprocessing import postprocess_predictions
from utils import resolve_device


def _convert_model_output(config, output):
    """Convert raw tensors to inference-ready probabilities/values."""
    if isinstance(output, dict) and "heads" in output:
        specs = get_head_specs(config)
        converted = {}
        for name, tensor in output["heads"].items():
            if str(specs[name]["task"]) == "classification":
                converted[name] = torch.softmax(tensor, dim=1)
            else:
                converted[name] = tensor.reshape(-1)
        return converted

    if isinstance(output, dict):
        output = output["logits"]
    if str(config.general.task) == "classification":
        return torch.softmax(output, dim=1)
    return output.reshape(-1)


def _append_batch(storage, value):
    if isinstance(value, dict):
        if storage is None:
            storage = {key: [] for key in value}
        for key, tensor in value.items():
            storage[key].append(tensor.detach().float().cpu().numpy())
        return storage
    if storage is None:
        storage = []
    storage.append(value.detach().float().cpu().numpy())
    return storage


def _concat_storage(storage):
    if isinstance(storage, dict):
        return {key: np.concatenate(values, axis=0) for key, values in storage.items()}
    return np.concatenate(storage, axis=0)


def _average_structures(values):
    first = values[0]
    if isinstance(first, dict):
        return {key: np.mean([value[key] for value in values], axis=0) for key in first}
    return np.mean(values, axis=0)


def _predict_one_model(config, checkpoint_path, loader, device):
    model = get_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    storage = None
    with torch.no_grad():
        for features in tqdm(loader, desc="Inference", leave=False) if config.logging.prints else loader:
            features = features.to(device, non_blocking=True)
            converted = _convert_model_output(config, model(features))
            storage = _append_batch(storage, converted)
    return _concat_storage(storage)


def _checkpoint_paths(config, checkpoint_root, folds=None):
    checkpoint_root = Path(checkpoint_root)
    if bool(config.split.all_data_train):
        paths = [checkpoint_root / "all_data" / "best.pt"]
    else:
        folds = [int(x) for x in (folds if folds is not None else config.split.folds_to_inference)]
        paths = [checkpoint_root / f"fold_{fold}" / "best.pt" for fold in folds]

    resolved = []
    for path in paths:
        if not path.exists():
            path = path.with_name("last.pt")
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        resolved.append(path)
    return resolved


def predict_features(config, features, checkpoint_root=None, folds=None):
    """Predict arbitrary already-loaded features with a fold ensemble."""
    loader = get_inference_loader(features, config)
    device = resolve_device(config)
    checkpoint_root = checkpoint_root or config.paths.path_to_fold_checkpoints
    predictions = [
        _predict_one_model(config, path, loader, device)
        for path in _checkpoint_paths(config, checkpoint_root, folds=folds)
    ]
    return _average_structures(predictions)


def get_inference_source(config):
    """Use final self-training stage when it exists; otherwise base folds."""
    if bool(config.strategies.self_training.enabled):
        manifest = Path(config.paths.path_to_self_training) / "manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            root = data.get("final_checkpoint_root")
            folds = data.get("folds")
            if root:
                return root, folds
    return config.paths.path_to_fold_checkpoints, None


def _finalize_predictions(config, predictions):
    if isinstance(predictions, dict):
        specs = get_head_specs(config)
        result = {}
        for name, values in predictions.items():
            if str(specs[name]["task"]) == "classification":
                result[name] = np.argmax(values, axis=1)
            else:
                result[name] = np.asarray(values).reshape(-1)
        return result
    if str(config.general.task) == "classification":
        return np.argmax(predictions, axis=1)
    return predictions


def _save_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(predictions, dict):
        # np.savez needs an .npz path. Keep configured stem but use correct extension.
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        np.savez(path, **predictions)
    else:
        np.save(path, predictions)
    return path


def inference(config):
    started = time.perf_counter()
    features = load_test_data(config)
    checkpoint_root, folds = get_inference_source(config)
    probabilities_or_values = predict_features(config, features, checkpoint_root=checkpoint_root, folds=folds)
    primary = probabilities_or_values[get_primary_head(config)] if isinstance(probabilities_or_values, dict) else probabilities_or_values
    task = get_primary_task(config)
    classes = list(range(get_primary_num_outputs(config))) if task == "classification" else None
    ids = _load_array(config.paths.path_to_test_ids) if config.paths.path_to_test_ids else np.arange(len(features))
    values = np.asarray(primary)
    if task == "classification" and len(classes) == 2:
        values = values[:, 1]
    path = Path(config.paths.path_to_predictions)
    save_predictions(path, ids, values, task, classes)
    if isinstance(probabilities_or_values, dict):
        np.savez(path.with_name("other_heads.npz"), **probabilities_or_values)
    final_predictions = _finalize_predictions(config, probabilities_or_values)
    if bool(config.postprocessing.enabled) and isinstance(final_predictions, dict):
        raise ValueError("Multi-head submission postprocessing needs a task-specific implementation")
    final_predictions = postprocess_predictions(final_predictions, config, probabilities=np.asarray(primary))
    if bool(config.postprocessing.enabled):
        pd.DataFrame({"id": ids, "prediction": final_predictions}).to_csv(path.with_name("submission.csv"), index=False)
    update_metadata(config.paths.path_to_metadata, inference_time_seconds=time.perf_counter()-started,
                    test_prediction_stage="base" if str(checkpoint_root) == str(config.paths.path_to_fold_checkpoints) else "self_training")
    if config.logging.prints:
        print(f"Predictions saved to: {path}")
    return final_predictions
