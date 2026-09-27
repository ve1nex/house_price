import numpy as np


def postprocess_predictions(predictions, config, probabilities=None):
    """Task-specific output hook.

    The base pipeline intentionally preserves predictions. A project may add
    thresholding, class-name mapping, clipping or submission/API formatting here.
    config.postprocessing.threshold is ready for binary/multilabel custom logic.
    """
    if not bool(config.postprocessing.enabled):
        return predictions
    if str(config.general.task) == "classification" and int(config.general.num_classes) == 2:
        if probabilities is None:
            raise ValueError("Binary threshold requires probability predictions")
        return (np.asarray(probabilities)[:, 1] >= float(config.postprocessing.threshold)).astype(int)
    raise NotImplementedError("Base postprocessing implements only binary thresholding; add task-specific logic")
