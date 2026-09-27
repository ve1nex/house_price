import numpy as np


def postprocess_predictions(predictions, config):
    """Optional project-specific output hook after model prediction."""
    values = np.asarray(predictions)
    if not bool(config.postprocessing.enabled):
        return values
    # Replace this exception with task-specific clipping/mapping if needed.
    raise NotImplementedError("Implement postprocess_predictions before enabling it")
