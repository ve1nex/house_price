from omegaconf import OmegaConf


def is_multi_head(config):
    return bool(config.strategies.multi_head.enabled)


def get_head_specs(config):
    if not is_multi_head(config):
        return {}
    return OmegaConf.to_container(config.strategies.multi_head.heads, resolve=True)


def get_primary_head(config):
    if not is_multi_head(config):
        return None
    name = str(config.strategies.multi_head.primary_head)
    heads = get_head_specs(config)
    if name not in heads:
        raise ValueError(f"multi_head.primary_head='{name}' is not present in multi_head.heads")
    return name


def get_primary_task(config):
    if not is_multi_head(config):
        return str(config.general.task)
    return str(get_head_specs(config)[get_primary_head(config)]["task"])


def get_primary_num_outputs(config):
    if not is_multi_head(config):
        return int(config.general.num_classes) if str(config.general.task) == "classification" else 1
    return int(get_head_specs(config)[get_primary_head(config)]["num_outputs"])


def get_metric_direction(config):
    if not is_multi_head(config):
        return str(config.metric.direction)
    return str(get_head_specs(config)[get_primary_head(config)]["metric_direction"])
