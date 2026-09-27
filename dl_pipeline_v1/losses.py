import torch
import torch.nn as nn
from omegaconf import OmegaConf

from metric_learning import get_metric_learning_loss
from multi_head import get_head_specs, get_primary_head, is_multi_head


def _build_torch_loss(name, params):
    if not hasattr(nn, name):
        raise ValueError(f"Unknown torch.nn loss: {name}")
    return getattr(nn, name)(**params)


class LossRouter:
    """Universal loss router for single-head, multi-head and metric learning."""

    def __init__(self, config):
        self.config = config
        self.multi_head = is_multi_head(config)

        if self.multi_head:
            self.head_specs = get_head_specs(config)
            self.head_losses = {
                name: _build_torch_loss(
                    str(spec["loss_name"]),
                    dict(spec.get("loss_params") or {}),
                )
                for name, spec in self.head_specs.items()
            }
            self.supervised_loss = None
        else:
            params = OmegaConf.to_container(config.loss.params, resolve=True)
            self.supervised_loss = _build_torch_loss(str(config.loss.name), params)
            self.head_specs = {}
            self.head_losses = {}

        self.metric_loss = get_metric_learning_loss(config) if bool(config.strategies.metric_learning.enabled) else None

    def __call__(self, outputs, labels):
        parts = {}

        if self.multi_head:
            if not isinstance(outputs, dict) or "heads" not in outputs:
                raise ValueError("Multi-head mode expects model outputs to contain outputs['heads']")
            supervised_total = 0.0
            for name, spec in self.head_specs.items():
                prediction = outputs["heads"][name]
                target = labels[name]
                if str(spec["task"]) == "regression":
                    prediction = prediction.reshape(-1)
                    target = target.float().reshape(-1)
                loss = self.head_losses[name](prediction, target)
                weight = float(spec.get("loss_weight", 1.0))
                supervised_total = supervised_total + weight * loss
                parts[f"loss/{name}"] = float(loss.detach().cpu())
        else:
            prediction = outputs["logits"] if isinstance(outputs, dict) else outputs
            target = labels
            if str(self.config.general.task) == "regression":
                prediction = prediction.reshape(-1)
                target = target.float().reshape(-1)
            supervised_total = self.supervised_loss(prediction, target)
            parts["loss/supervised"] = float(supervised_total.detach().cpu())

        total = (float(self.config.strategies.metric_learning.supervised_loss_weight) * supervised_total
                 if self.metric_loss is not None else supervised_total)

        if self.metric_loss is not None:
            if not isinstance(outputs, dict) or "embeddings" not in outputs:
                raise ValueError("metric_learning.enabled=True expects model outputs['embeddings']")
            if self.multi_head:
                label_head = self.config.strategies.metric_learning.label_head
                label_head = str(label_head) if label_head else get_primary_head(self.config)
                metric_labels = labels[label_head]
            else:
                metric_labels = labels
            metric_loss = self.metric_loss(outputs["embeddings"], metric_labels)
            total = total + float(self.config.strategies.metric_learning.metric_loss_weight) * metric_loss
            parts["loss/metric"] = float(metric_loss.detach().cpu())

        parts["loss/total"] = float(total.detach().cpu())
        return total, parts


def get_loss(config):
    return LossRouter(config)
