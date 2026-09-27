import math

import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """Yield balanced class batches for metric learning.

    Each batch contains classes_per_batch classes and samples_per_class samples
    from every chosen class. Samples are re-used across an epoch if a class is small.
    """

    def __init__(self, labels, classes_per_batch, samples_per_class, seed=42):
        self.labels = np.asarray(labels).reshape(-1)
        self.classes = np.unique(self.labels)
        self.classes_per_batch = min(int(classes_per_batch), len(self.classes))
        self.samples_per_class = int(samples_per_class)
        if self.classes_per_batch < 2:
            raise ValueError("Metric learning needs at least two classes per batch")
        if self.samples_per_class < 2:
            raise ValueError("Metric learning needs at least two samples per class")
        self.seed = int(seed)
        self.epoch = 0
        self.indices_by_class = {
            cls: np.where(self.labels == cls)[0] for cls in self.classes
        }
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.num_batches = max(1, math.ceil(len(self.labels) / self.batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            chosen_classes = rng.choice(self.classes, size=self.classes_per_batch, replace=False)
            batch = []
            for cls in chosen_classes:
                pool = self.indices_by_class[cls]
                replace = len(pool) < self.samples_per_class
                batch.extend(rng.choice(pool, size=self.samples_per_class, replace=replace).tolist())
            rng.shuffle(batch)
            yield batch


def get_balanced_batch_sampler(labels, config, seed=None):
    if not bool(config.strategies.metric_learning.enabled) or not bool(config.strategies.metric_learning.balanced_batches):
        return None
    return BalancedBatchSampler(
        labels,
        classes_per_batch=int(config.strategies.metric_learning.classes_per_batch),
        samples_per_class=int(config.strategies.metric_learning.samples_per_class),
        seed=int(config.general.seed) if seed is None else int(seed),
    )
