def generate_dataset(config):
    """Optional offline dataset-generation hook.

    The mentor's original implementation generated features specifically from
    sign-language landmarks. That task-specific code is intentionally removed.

    Use this file only when a future project needs an offline conversion step,
    for example raw files -> cached tensors. The training pipeline does not call
    this function automatically.
    """
    raise NotImplementedError(
        "Implement dataset generation only for a project that actually needs it."
    )
