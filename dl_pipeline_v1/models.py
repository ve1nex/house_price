import math
import torch.nn as nn


def _activation(name):
    """Resolve a torch activation by its configured name."""
    if not hasattr(nn, name):
        raise ValueError(f"Unknown activation: {name}")
    return getattr(nn, name)


class MLPBackbone(nn.Module):
    """Retain the original layer names so saved House Prices weights still load."""

    def __init__(self, input_shape, hidden_dims, dropout=0.0, activation="ReLU"):
        super().__init__()
        layers = [nn.Flatten()]
        current = int(math.prod(input_shape))
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.extend(
                [
                    nn.Linear(current, hidden),
                    _activation(activation)(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current = hidden
        self.network = nn.Sequential(*layers)
        self.feature_dim = current

    def forward(self, x):
        return self.network(x)


class PredictionHead(nn.Module):
    """Predict the scalar transformed sale price."""

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(int(input_dim), 1))

    def forward(self, x):
        return self.network(x)


class UniversalDLModel(nn.Module):
    """Keep the backbone/head interface used by the existing checkpoints."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = PredictionHead(backbone.feature_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


def get_model(config):
    """Build the configured MLP without changing the training loop."""
    if str(config.model.name) != "MLP":
        raise ValueError("House Prices DL supports MLP")
    dropout = (
        float(config.regularization.dropout.p)
        if config.regularization.enabled and config.regularization.dropout.enabled
        else 0.0
    )
    return UniversalDLModel(
        MLPBackbone(
            list(config.model.input_shape),
            list(config.model.params.hidden_dims),
            dropout=dropout,
            activation=str(config.model.params.activation),
        )
    )


def count_parameters(model):
    """Return total and trainable parameter counts."""
    return sum(p.numel() for p in model.parameters()), sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
