import torch
from omegaconf import OmegaConf


def _head_parameters(model):
    params = []
    if getattr(model, "head", None) is not None:
        params.extend(list(model.head.parameters()))
    if getattr(model, "heads", None) is not None:
        params.extend(list(model.heads.parameters()))
    if getattr(model, "embedding_head", None) is not None:
        params.extend(list(model.embedding_head.parameters()))
    return params


def get_optimizer(config, model):
    """Create standard optimizer, with optional discriminative LR for fine-tuning."""
    name = str(config.optimizer.name)
    if not hasattr(torch.optim, name):
        raise ValueError(f"Unknown torch optimizer: {name}")
    optimizer_cls = getattr(torch.optim, name)
    params = OmegaConf.to_container(config.optimizer.params, resolve=True)
    params["weight_decay"] = float(config.regularization.weight_decay.value) if bool(config.regularization.enabled) and bool(config.regularization.weight_decay.enabled) else 0.0
    if bool(config.optimization.enabled) and bool(config.optimization.speed.fused_optimizer.enabled):
        if name not in {"Adam", "AdamW"} or not torch.cuda.is_available():
            raise ValueError("fused_optimizer requires Adam/AdamW on CUDA")
        params["fused"] = True

    if not bool(config.strategies.finetuning.enabled):
        return optimizer_cls(model.parameters(), **params)

    if not hasattr(model, "backbone"):
        raise AttributeError("Fine-tuning optimizer requires model.backbone")

    base_params = dict(params)
    base_params.pop("lr", None)
    parameter_groups = [
        {"params": list(model.backbone.parameters()), "lr": float(config.strategies.finetuning.backbone_lr)},
        {"params": _head_parameters(model), "lr": float(config.strategies.finetuning.head_lr)},
    ]
    return optimizer_cls(parameter_groups, **base_params)
