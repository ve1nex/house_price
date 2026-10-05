import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from artifact_io import update_metadata
from data import load_csv, prepare_dataframe
from features import feature_engineering


def inference(config):
    """Load the saved pipeline and write prices in the Kaggle submission format."""
    started = time.perf_counter()
    pipeline = joblib.load(Path(config.paths.path_to_final_model))
    df = feature_engineering(
        prepare_dataframe(load_csv(config.paths.path_to_test_dataset), config), config
    )
    ids = df[str(config.data.id_column)].to_numpy()
    values = pipeline.predict(df.drop(columns=[str(config.data.id_column)]))
    if str(config.data.target_transform) == "log1p":
        values = np.expm1(values)
    elif str(config.data.target_transform) != "none":
        raise ValueError("Unknown target_transform")
    output = pd.DataFrame(
        {str(config.data.id_column): ids, str(config.data.target): values}
    )
    path = Path(config.paths.path_to_predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    update_metadata(
        config.paths.path_to_metadata,
        inference_time_seconds=time.perf_counter() - started,
    )
    if config.logging.prints:
        print(f"Saved predictions: {path}")
    return output
