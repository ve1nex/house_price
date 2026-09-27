def preprocess_sample(x, config, is_train: bool):
    """Task-specific preprocessing hook for one sample.

    Examples for future projects: normalization, tokenization, resizing, sequence
    cropping/padding, spectrogram creation, keypoint selection, etc.
    The base implementation is intentionally a no-op.
    """
    if not bool(config.preprocessing.enabled):
        return x
    raise NotImplementedError("preprocessing.enabled requires task-specific transforms in preprocessing.py")
