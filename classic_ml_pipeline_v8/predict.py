from pathlib import Path
import time
import json

import joblib
import pandas as pd
import numpy as np

from data import load_csv, prepare_dataframe
from features import feature_engineering
from postprocessing import postprocess_predictions
from utils import reduce_mem_usage
from artifact_io import save_predictions, update_metadata


def inference(config):
    """Load the saved final pipeline and predict the test dataset."""
    model_path = Path(config.paths.path_to_final_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    started = time.perf_counter()
    pipeline = joblib.load(model_path)
    df = load_csv(config.paths.path_to_test_dataset)

    # --- Memory optimization ---
    df = reduce_mem_usage(
        df,
        enabled=bool(config.data.reduce_memory),
        verbose=bool(config.logging.prints),
    )

    df = prepare_dataframe(df, config)

    # --- Feature engineering ---
    # The same task-specific hook used during training must also run at inference.
    df = feature_engineering(df, config)

    # ID can be kept in the output but should not be passed into the model.
    if config.data.id_column and config.data.id_column not in df.columns:
        raise ValueError(f"Configured data.id_column '{config.data.id_column}' is missing from test data")
    ids = df[config.data.id_column].to_numpy() if config.data.id_column else df.index.to_numpy()

    feature_drop = []
    if config.data.id_column is not None and config.data.id_column in df.columns:
        feature_drop.append(config.data.id_column)
    if config.split.group_column is not None and config.split.group_column in df.columns:
        feature_drop.append(config.split.group_column)

    X_test = df.drop(columns=list(dict.fromkeys(feature_drop)), errors="ignore")
    if str(config.general.task) == "classification":
        classes = list(pipeline.classes_)
        probabilities = pipeline.predict_proba(X_test)
        predictions = probabilities[:, 1] if len(classes) == 2 else probabilities
    else:
        classes = None
        predictions = pipeline.predict(X_test)

    # --- Prediction post-processing ---
    # Empty by default; task-specific clipping/thresholds can be added later.
    predictions = postprocess_predictions(predictions, config)

    output_path = Path(config.paths.path_to_predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = save_predictions(output_path, ids, predictions, str(config.general.task), classes)
    if str(config.general.task) == "classification" and len(classes) == 2 and bool(config.optimization.enabled) and bool(config.optimization.prediction.threshold_tuning.enabled):
        threshold_path = Path(config.paths.path_to_checkpoints) / "threshold.json"
        if not threshold_path.exists():
            raise FileNotFoundError("Threshold tuning enabled but threshold.json is missing; train before inference")
        threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
        labels = np.where(np.asarray(predictions) >= threshold, classes[1], classes[0])
        pd.DataFrame({"id": ids, "prediction": labels}).to_csv(output_path.with_name("submission.csv"), index=False)
    update_metadata(config.paths.path_to_metadata, inference_time_seconds=time.perf_counter()-started)

    if config.logging.prints:
        print(f"Saved predictions: {output_path}")

    return output
