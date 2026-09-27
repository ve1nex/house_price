def apply_augmentations(x, config):
    """Task-specific augmentation hook.

    Keep this stage in every DL project, but put only the concrete transforms of
    the current task here (image flips/crops, audio transforms, sequence noise,
    MixUp, etc.). The reusable base pipeline intentionally does nothing.
    """
    if not bool(config.augmentations.enabled):
        return x
    raise NotImplementedError("augmentations.enabled requires task-specific transforms in augmentations.py")
