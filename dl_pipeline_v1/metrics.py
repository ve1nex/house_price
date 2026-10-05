import numpy as np
import sklearn.metrics


def get_metric(config, y_true, raw_outputs):
    """Calculate the metric in the same transformed target space as training."""
    name = str(config.metric.name)
    if not hasattr(sklearn.metrics, name):
        raise ValueError(f"Unknown metric: {name}")
    return float(
        getattr(sklearn.metrics, name)(
            np.asarray(y_true).reshape(-1),
            np.asarray(raw_outputs).reshape(-1),
            **dict(config.metric.params),
        )
    )


def is_improvement(config, current, best):
    """Compare a new fold metric with the checkpoint selection metric."""
    if best is None:
        return True
    if str(config.metric.direction) == "minimize":
        return current < best
    if str(config.metric.direction) == "maximize":
        return current > best
    raise ValueError("metric.direction must be minimize or maximize")
