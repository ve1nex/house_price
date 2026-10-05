from omegaconf import OmegaConf

from checks import check_training_data
from config import config
from data import load_training_data
from predict import inference
from prepare_house_prices import load_raw_data
from train import train
from utils import (
    prepare_experiment,
    save_config_snapshot,
    save_dataset_metadata,
    set_seed,
)


def fit(config):
    """Load raw tables, optionally tune the MLP, and run fold-local training."""
    set_seed(
        int(config.general.seed),
        deterministic=bool(config.reproducibility.deterministic),
    )
    features, labels, groups, fold_ids = load_training_data(config)
    check_training_data(features, labels, config)
    prepare_experiment(config)
    save_dataset_metadata(config, features, labels, groups, fold_ids)
    if config.tuning.enabled:
        from tuning import apply_best_params, run_tuning

        best_params, _ = run_tuning(
            config, features, labels, groups=groups, fold_ids=fold_ids
        )
        apply_best_params(config, best_params)
    save_config_snapshot(config)
    results = train(
        config,
        features,
        labels,
        groups=groups,
        fold_ids=fold_ids,
        ids=load_raw_data(config, train=True)[2],
    )
    inference(config)
    return results


def main():
    """Dispatch the configured training or saved-model inference mode."""
    if config.general.mode == "train":
        return fit(config)
    if config.general.mode == "inference":
        return inference(config)
    raise ValueError("general.mode must be train or inference")


if __name__ == "__main__":
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    main()
