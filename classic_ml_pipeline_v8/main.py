from omegaconf import OmegaConf

from checks import check_data_leakage
from config import config
from data import load_csv, prepare_dataframe, split_features_target
from features import feature_engineering
from predict import inference
from train import train
from utils import (
    dataframe_info,
    prepare_experiment,
    save_dataset_metadata,
    save_config_snapshot,
    set_seed,
)


def fit(config):
    """Prepare House Prices data and run the configured cross-validation."""
    set_seed(int(config.general.seed))
    df = load_csv(config.paths.path_to_train_dataset)
    raw_info = dataframe_info(df)
    prepare_experiment(config)
    df = prepare_dataframe(df, config)
    if config.data.drop_train_ids:
        df = df.loc[
            ~df[config.data.id_column].isin(config.data.drop_train_ids)
        ].reset_index(drop=True)
    df = feature_engineering(df, config)
    prepared_info = dataframe_info(df)
    save_dataset_metadata(config, raw_info, prepared_info, prepared_info)
    X, y = split_features_target(df, config)
    ids = df[config.data.id_column].to_numpy()
    check_data_leakage(X, y, config)
    if bool(config.tuning.enabled):
        from tuning import apply_best_params, run_tuning

        best_params, study = run_tuning(config, X, y)
        apply_best_params(config, best_params)
        if config.logging.prints:
            print(f"Best tuning CV: {study.best_value:.6f}")
    save_config_snapshot(config)
    return train(config, X, y, ids=ids)


def main():
    """Run training or inference using config.py."""
    if config.general.mode == "train":
        return fit(config)
    if config.general.mode == "inference":
        return inference(config)
    raise ValueError("general.mode must be 'train' or 'inference'")


if __name__ == "__main__":
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    main()
