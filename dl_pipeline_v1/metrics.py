import numpy as np
import sklearn.metrics

from multi_head import get_head_specs, get_metric_direction, get_primary_head, is_multi_head

PROBABILITY_METRICS = {"roc_auc_score", "average_precision_score", "log_loss", "brier_score_loss"}


def _softmax(logits):
    values = np.asarray(logits, dtype=float)
    values -= values.max(axis=1, keepdims=True)
    e = np.exp(values)
    return e / e.sum(axis=1, keepdims=True)


def _prediction_for_task(task, outputs):
    outputs = np.asarray(outputs)
    if str(task) == "classification":
        return np.argmax(outputs, axis=1)
    return outputs.reshape(-1)


def get_primary_raw_outputs(config, raw_outputs):
    if is_multi_head(config):
        return raw_outputs["heads"][get_primary_head(config)]
    if isinstance(raw_outputs, dict):
        return raw_outputs["logits"]
    return raw_outputs


def get_primary_targets(config, targets):
    if is_multi_head(config):
        return targets[get_primary_head(config)]
    return targets


def outputs_to_predictions(config, outputs):
    raw = get_primary_raw_outputs(config, outputs)
    if is_multi_head(config):
        spec = get_head_specs(config)[get_primary_head(config)]
        return _prediction_for_task(spec["task"], raw)
    return _prediction_for_task(config.general.task, raw)


def get_metric(config, y_true, raw_outputs):
    if is_multi_head(config):
        head = get_primary_head(config)
        spec = get_head_specs(config)[head]
        name = str(spec["metric_name"])
        params = dict(spec.get("metric_params") or {})
        true_values = np.asarray(y_true[head])
        predictions = _prediction_for_task(spec["task"], raw_outputs["heads"][head])
    else:
        name = str(config.metric.name)
        params = dict(config.metric.params)
        true_values = np.asarray(y_true)
        predictions = outputs_to_predictions(config, raw_outputs)

    if not hasattr(sklearn.metrics, name):
        raise ValueError(f"Unknown sklearn metric: {name}")
    if name in PROBABILITY_METRICS:
        if is_multi_head(config):
            raw = raw_outputs["heads"][head]
            task = str(spec["task"])
        else:
            raw = get_primary_raw_outputs(config, raw_outputs)
            task = str(config.general.task)
        if task != "classification":
            raise ValueError(f"{name} requires classification probabilities")
        probabilities = _softmax(raw)
        prediction = probabilities if name == "log_loss" or probabilities.shape[1] > 2 else probabilities[:, 1]
        if name == "log_loss":
            params.setdefault("labels", list(range(probabilities.shape[1])))
        if name == "roc_auc_score" and probabilities.shape[1] > 2:
            params.setdefault("multi_class", "ovr")
            params.setdefault("labels", list(range(probabilities.shape[1])))
        return float(getattr(sklearn.metrics, name)(true_values, prediction, **params))
    return float(getattr(sklearn.metrics, name)(true_values, predictions, **params))


def is_improvement(config, current, best):
    direction = get_metric_direction(config)
    if best is None:
        return True
    if direction == "maximize":
        return current > best
    if direction == "minimize":
        return current < best
    raise ValueError("Metric direction must be 'maximize' or 'minimize'")
