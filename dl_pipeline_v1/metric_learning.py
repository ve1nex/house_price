import torch
import torch.nn as nn
import torch.nn.functional as F

from hard_negative_mining import select_positive_negative


class BatchTripletLoss(nn.Module):
    """Triplet loss inside a batch with optional hard-negative mining."""

    def __init__(self, margin=0.3, hard_negative=False):
        super().__init__()
        self.margin = float(margin)
        self.hard_negative = bool(hard_negative)

    def forward(self, embeddings, labels):
        if embeddings.ndim != 2:
            raise ValueError("Metric-learning embeddings must have shape [batch, embedding_dim]")
        labels = labels.reshape(-1)
        distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)
        positives, negatives = select_positive_negative(
            distances, labels, hard_negative=self.hard_negative
        )
        if positives is None:
            # Keep graph connected when a batch contains no valid positive/negative pairs.
            return embeddings.sum() * 0.0
        return F.relu(positives - negatives + self.margin).mean()


def get_metric_learning_loss(config):
    if bool(config.strategies.hard_negative_mining.enabled) and str(config.strategies.hard_negative_mining.strategy) != "batch_hard":
        raise ValueError("Only hard_negative_mining.strategy='batch_hard' is implemented")
    return BatchTripletLoss(
        margin=float(config.strategies.metric_learning.margin),
        hard_negative=bool(config.strategies.hard_negative_mining.enabled),
    )
