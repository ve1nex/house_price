import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_head import get_head_specs, is_multi_head


def _activation(name):
    if not hasattr(nn, name):
        raise ValueError(f"Unknown torch activation: {name}")
    return getattr(nn, name)


class MLPBackbone(nn.Module):
    def __init__(self, input_shape, hidden_dims, dropout=0.0, activation="ReLU"):
        super().__init__()
        input_dim = int(math.prod(input_shape))
        activation_cls = _activation(activation)
        layers = [nn.Flatten()]
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                activation_cls(),
                nn.Dropout(float(dropout)),
            ])
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.feature_dim = current_dim

    def forward(self, x):
        return self.network(x)


class TransformerBackbone(nn.Module):
    """Generic sequence backbone for inputs shaped [batch, sequence, features]."""

    def __init__(
        self,
        input_shape,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError("TransformerMLP expects model.input_shape=[sequence_length, feature_dim]")
        sequence_length, input_dim = int(input_shape[0]), int(input_shape[1])
        d_model = int(d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.feature_dim = d_model

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("TransformerMLP input must have shape [batch, sequence, features]")
        if x.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("Input sequence is longer than model.input_shape sequence length")
        x = self.input_projection(x)
        x = x + self.position_embedding[:, : x.shape[1]]
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)


class PredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.0, activation="ReLU"):
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        activation_cls = _activation(activation)
        layers = []
        current = int(input_dim)
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.extend([nn.Linear(current, hidden), activation_cls(), nn.Dropout(float(dropout))])
            current = hidden
        layers.append(nn.Linear(current, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class UniversalDLModel(nn.Module):
    """One reusable output router: single head, multi-head and/or embeddings."""

    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(backbone.feature_dim)
        self.multi_head = is_multi_head(config)
        self.metric_learning = bool(config.strategies.metric_learning.enabled)

        use_mlp_head = str(config.model.name) == "TransformerMLP"
        head_hidden = list(config.model.params.head_hidden_dims) if use_mlp_head else []
        head_dropout = (float(config.regularization.dropout.p) if bool(config.regularization.enabled) and bool(config.regularization.dropout.enabled) and use_mlp_head else 0.0)
        head_activation = str(config.model.params.activation)

        if self.multi_head:
            specs = get_head_specs(config)
            self.heads = nn.ModuleDict({
                name: PredictionHead(
                    self.feature_dim,
                    int(spec["num_outputs"]),
                    hidden_dims=head_hidden,
                    dropout=head_dropout,
                    activation=head_activation,
                )
                for name, spec in specs.items()
            })
            self.head = None
        else:
            output_dim = int(config.general.num_classes) if str(config.general.task) == "classification" else 1
            self.head = PredictionHead(
                self.feature_dim,
                output_dim,
                hidden_dims=head_hidden,
                dropout=head_dropout,
                activation=head_activation,
            )
            self.heads = None

        if self.metric_learning:
            self.embedding_head = nn.Linear(self.feature_dim, int(config.strategies.metric_learning.embedding_dim))
        else:
            self.embedding_head = None

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.forward_features(x)

        if self.multi_head:
            head_outputs = {name: head(features) for name, head in self.heads.items()}
            output = {"heads": head_outputs}
        else:
            logits = self.head(features)
            if not self.metric_learning:
                return logits
            output = {"logits": logits}

        if self.metric_learning:
            embeddings = self.embedding_head(features)
            if bool(getattr(self._metric_cfg, "normalize_embeddings", True)):
                embeddings = F.normalize(embeddings, p=2, dim=1)
            output["embeddings"] = embeddings
        return output

    @property
    def _metric_cfg(self):
        # Config is injected after construction to avoid keeping OmegaConf as a Module child.
        return self.__dict__["_config_metric_learning"]

    def set_metric_config(self, metric_cfg):
        self.__dict__["_config_metric_learning"] = metric_cfg
        return self


def _build_backbone(config):
    name = str(config.model.name)
    params = config.model.params
    dropout = float(config.regularization.dropout.p) if bool(config.regularization.enabled) and bool(config.regularization.dropout.enabled) else 0.0

    if name == "MLP":
        return MLPBackbone(
            input_shape=list(config.model.input_shape),
            hidden_dims=list(params.hidden_dims),
            dropout=dropout,
            activation=str(params.activation),
        )

    if name == "TransformerMLP":
        return TransformerBackbone(
            input_shape=list(config.model.input_shape),
            d_model=int(params.d_model),
            nhead=int(params.nhead),
            num_layers=int(params.num_layers),
            dim_feedforward=int(params.dim_feedforward),
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown model: {name}. Add a backbone to models.py and keep train.py unchanged."
    )


def get_model(config):
    model = UniversalDLModel(_build_backbone(config), config)
    return model.set_metric_config(config.strategies.metric_learning)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_batchnorms(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
