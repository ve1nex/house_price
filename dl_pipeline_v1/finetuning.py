from pathlib import Path

import torch


def _extract_state_dict(value):
    if isinstance(value, dict) and "model" in value and isinstance(value["model"], dict):
        return value["model"]
    if isinstance(value, dict) and "state_dict" in value and isinstance(value["state_dict"], dict):
        return value["state_dict"]
    if isinstance(value, dict):
        return value
    raise ValueError("Unsupported pretrained checkpoint format")


def load_pretrained_weights(model, config, map_location="cpu"):
    """Load reusable pretrained weights.

    For library-native pretrained models (torchvision/HuggingFace/etc.), construct
    the pretrained backbone inside models.py and use finetuning.source=model_builtin.
    """
    if not bool(config.strategies.finetuning.enabled):
        return None
    if str(config.strategies.finetuning.source) == "model_builtin":
        if str(config.model.name) in {"MLP", "TransformerMLP"}:
            raise ValueError("Built-in MLP/TransformerMLP have no pretrained weights; implement a pretrained backbone in models.py")
        return None

    path = config.strategies.finetuning.checkpoint_path
    if not path:
        raise ValueError("finetuning.enabled=True with source=checkpoint requires checkpoint_path")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    value = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = _extract_state_dict(value)
    strict = bool(config.strategies.finetuning.strict_load)
    if strict:
        return model.load_state_dict(state_dict, strict=True)

    # Transfer learning often changes the final head size. Keep only keys that
    # exist in the new model and have the same tensor shape.
    current = model.state_dict()
    compatible = {
        key: tensor for key, tensor in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(tensor.shape)
    }
    current.update(compatible)
    result = model.load_state_dict(current, strict=False)
    return {
        "loaded_keys": len(compatible),
        "skipped_keys": len(state_dict) - len(compatible),
        "result": result,
    }


def freeze_backbone(model):
    if not hasattr(model, "backbone"):
        raise AttributeError("Fine-tuning requires the model to expose .backbone")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def unfreeze_backbone(model):
    if not hasattr(model, "backbone"):
        raise AttributeError("Fine-tuning requires the model to expose .backbone")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = True


def setup_finetuning(model, config, map_location="cpu"):
    if not bool(config.strategies.finetuning.enabled):
        return
    load_pretrained_weights(model, config, map_location=map_location)
    if bool(config.strategies.finetuning.freeze_backbone) and int(config.strategies.finetuning.unfreeze_after_epoch) > 0:
        freeze_backbone(model)


def maybe_unfreeze_backbone(model, config, epoch_index):
    """Called before each epoch; epoch_index is zero-based."""
    if not bool(config.strategies.finetuning.enabled):
        return False
    if not bool(config.strategies.finetuning.freeze_backbone):
        return False
    target = int(config.strategies.finetuning.unfreeze_after_epoch)
    if target <= 0 or epoch_index < target:
        return False
    if all(parameter.requires_grad for parameter in model.backbone.parameters()):
        return False
    unfreeze_backbone(model)
    return True
