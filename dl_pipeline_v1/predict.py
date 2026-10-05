import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from artifact_io import update_metadata
from data import _load_array, get_inference_loader, load_test_data
from models import get_model
from prepare_house_prices import load_raw_data
from utils import resolve_device


def predict_features(config, features, checkpoint_root=None, folds=None):
    """Average fold predictions after applying each checkpoint's own preprocessor."""
    device = resolve_device(config)
    root = Path(checkpoint_root or config.paths.path_to_fold_checkpoints)
    folds = [
        int(f)
        for f in (folds if folds is not None else config.split.folds_to_inference)
    ]
    if not folds:
        raise ValueError("Select at least one inference fold")
    predictions = []
    for fold in folds:
        directory = root / f"fold_{fold}"
        path = directory / "best.pt"
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        fold_config = copy.deepcopy(config)
        if checkpoint.get("input_shape") is not None:
            fold_config.model.input_shape = checkpoint["input_shape"]
        preprocessor_path = directory / "preprocessor.joblib"
        if checkpoint.get("preprocessing") == "per_fold":
            if not preprocessor_path.exists():
                raise FileNotFoundError(
                    f"Fold preprocessor is missing: {preprocessor_path}"
                )
            matrix = joblib.load(preprocessor_path).transform(features)
        else:
            # Original weights were trained on the supplied global preprocessing arrays.
            matrix = (
                _load_array(config.paths.path_to_test_features)
                if isinstance(features, pd.DataFrame)
                else features
            )
        model = get_model(fold_config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        batches = []
        with torch.no_grad():
            for x in get_inference_loader(
                np.asarray(matrix, dtype=np.float32), fold_config
            ):
                batches.append(model(x.to(device)).reshape(-1).cpu().numpy())
        predictions.append(np.concatenate(batches))
    return np.mean(predictions, axis=0)


def inference(config):
    """Convert the averaged transformed target back to prices and save the submission."""
    started = time.perf_counter()
    features = load_test_data(config)
    values = predict_features(config, features)
    if str(config.data.target_transform) == "log1p":
        values = np.expm1(values)
    elif str(config.data.target_transform) != "none":
        raise ValueError("Unknown target_transform")
    ids = load_raw_data(config, train=False)[2]
    if len(values) != len(ids):
        raise ValueError("Prediction count differs from test IDs")
    output = pd.DataFrame({"Id": ids, "SalePrice": values})
    path = Path(config.paths.path_to_predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    update_metadata(
        config.paths.path_to_metadata,
        inference_time_seconds=time.perf_counter() - started,
        test_prediction_stage="base",
    )
    if config.logging.prints:
        print(f"Saved predictions: {path}")
    return output
