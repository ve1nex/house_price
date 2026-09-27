from pathlib import Path

import torch
import torch.nn as nn

from checkpointing import load_checkpoint
from models import get_model
from multi_head import get_head_specs, is_multi_head
from utils import resolve_device


class ExportWrapper(nn.Module):
    """Flatten training-only dict outputs into deployment-friendly tensors/tuples."""

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.multi_head = is_multi_head(config)
        self.head_names = list(get_head_specs(config).keys()) if self.multi_head else []

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict) and "heads" in output:
            values = tuple(output["heads"][name] for name in self.head_names)
            return values[0] if len(values) == 1 else values
        if isinstance(output, dict):
            return output["logits"]
        return output


def export_model(config, checkpoint_path):
    if not bool(config.conversion.enabled):
        return []

    output_dir = Path(config.paths.path_to_exports)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config)

    model = get_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    wrapper = ExportWrapper(model.eval(), config).to(device).eval()

    sample = torch.randn(2, *list(config.model.input_shape), device=device)
    exported = []
    output_names = list(get_head_specs(config).keys()) if is_multi_head(config) else ["outputs"]

    for fmt in config.conversion.formats:
        fmt = str(fmt).lower()
        if fmt == "torch_export":
            exported_program = torch.export.export(wrapper, (sample,), dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}})
            path = output_dir / "model.pt2"
            torch.export.save(exported_program, str(path))
            exported.append(str(path))
        elif fmt == "onnx":
            path = output_dir / "model.onnx"
            torch.onnx.export(
                wrapper,
                sample,
                str(path),
                opset_version=int(config.conversion.opset_version),
                input_names=["inputs"],
                output_names=output_names,
                dynamo=True,
                dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
            )
            exported.append(str(path))
        else:
            raise ValueError(f"Unknown export format: {fmt}")
    return exported
