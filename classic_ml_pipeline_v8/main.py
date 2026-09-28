from checks import check_data_leakage
from config import config
from data import get_groups, load_csv, prepare_dataframe, split_features_target
from features import feature_engineering
from notifier import send_telegram
from predict import inference
from train import train
from tuning import apply_best_params, run_tuning
from tracking import finish_tracking, log_experiment_artifacts, log_summary, start_tracking
from utils import (
    dataframe_info,
    ensure_directories,
    prepare_experiment,
    reduce_mem_usage,
    save_dataset_metadata,
    save_config_snapshot,
    set_seed,
)


def fit(config):
    # --- Technical setup ---
    # Reproducibility seed. Experiment folders are prepared only after the dataset
    # is successfully found, so a wrong path does not create a broken experiment.
    set_seed(config.general.seed)

    # --- Data loading ---
    df = load_csv(config.paths.path_to_train_dataset)
    raw_info = dataframe_info(df)

    # --- Experiment reproducibility ---
    # Protect old runs, create folders, save resolved config and environment versions.
    prepare_experiment(config)

    # --- Memory optimization ---
    # Technical utility: reduce DataFrame RAM usage without task-specific logic.
    df = reduce_mem_usage(
        df,
        enabled=bool(config.data.reduce_memory),
        verbose=bool(config.logging.prints),
    )

    # --- Basic preprocessing ---
    # Universal cleanup only: columns explicitly listed in config are removed.
    df = prepare_dataframe(df, config)
    drop_ids = list(config.data.drop_train_ids)

    if drop_ids:
        id_col = str(config.data.id_column)

        before = len(df)
        df = df[~df[id_col].isin(drop_ids)].reset_index(drop=True)

        if config.logging.prints:
            print(f"Dropped train rows: {before - len(df)} | IDs: {drop_ids}")
    # --- Feature engineering ---
    # Task-specific stage is intentionally kept as a placeholder.
    # Put the concrete features of a new task in features.py.
    df = feature_engineering(df, config)
    prepared_info = dataframe_info(df)

    # --- Debug mode ---
    # Fast run on a small sample before a full experiment.
    if config.training.debug:
        df = df.sample(
            min(int(config.training.debug_n_rows), len(df)),
            random_state=int(config.general.seed),
        ).reset_index(drop=True)

    training_info = dataframe_info(df)
    save_dataset_metadata(config, raw_info, prepared_info, training_info)

    # --- Remote experiment tracking ---
    # Local files remain the source of truth; W&B is an optional dashboard.
    start_tracking(config)

    # --- Features / target / groups ---
    # Groups are used only by GroupKFold / StratifiedGroupKFold.
    groups = get_groups(df, config)
    X, y = split_features_target(df, config)
    if config.data.id_column and config.data.id_column not in df.columns:
        raise ValueError(f"Configured data.id_column '{config.data.id_column}' is absent after preparation")
    ids = df[config.data.id_column].to_numpy() if config.data.id_column else df.index.to_numpy()
    if str(config.general.task) == "classification":
        config.general.num_classes = int(y.nunique())
    save_config_snapshot(config)

    # --- Leakage check ---
    # Universal checks stay here; project-specific leakage rules can be added in checks.py.
    check_data_leakage(X, y, config)

    # --- Hyperparameter tuning ---
    # Optional Optuna stage: search -> best params -> normal full training.
    if bool(config.tuning.enabled):
        best_params, study = run_tuning(config, X, y, groups=groups)
        apply_best_params(config, best_params)
        save_config_snapshot(config)
        log_summary(
            config,
            {
                "tuning_best_value": float(study.best_value),
                "tuning_best_params": dict(best_params),
            },
        )
        send_telegram(
            f"{config.general.experiment_name} tuning finished\n"
            f"Best CV: {study.best_value:.6f}\n"
            f"Best params: {best_params}",
            config,
        )

    # --- Solver / training ---
    # Cross-validation, metric calculation, checkpoints and final model.
    train(config, X, y, groups=groups, ids=ids)
    log_experiment_artifacts(config)
    finish_tracking(config, exit_code=0)


def main():
    # --- Train / inference ---
    try:
        if config.general.mode == "train":
            fit(config)
        elif config.general.mode == "inference":
            ensure_directories(config)
            send_telegram(
                f"{config.general.experiment_name} inference started",
                config,
            )
            output = inference(config)
            send_telegram(
                f"{config.general.experiment_name} inference finished\n"
                f"Rows: {len(output)}\n"
                f"Saved: {config.paths.path_to_predictions}",
                config,
            )
        else:
            raise ValueError("config.general.mode must be 'train' or 'inference'")
    except Exception as error:
        finish_tracking(config, exit_code=1)
        send_telegram(
            f"{config.general.experiment_name} FAILED\n"
            f"Mode: {config.general.mode}\n"
            f"Error: {type(error).__name__}: {error}",
            config,
        )
        raise


if __name__ == "__main__":
    # --- Run ---
    main()
