from checks import check_training_data
from config import config
from data import load_training_data
from model_conversion import export_model
from predict import inference
from notifier import TelegramNotifier
from self_training import run_self_training
from train import train
from tuning import apply_best_params, run_tuning
from utils import ensure_directories, prepare_experiment, save_config_snapshot, save_dataset_metadata, set_seed
from pathlib import Path
from data import _load_array


def fit(config):
    # --- Technical setup ---
    set_seed(int(config.general.seed), deterministic=bool(config.reproducibility.deterministic))

    # --- Data loading ---
    features, labels, groups, fold_ids = load_training_data(config)

    # --- Experiment reproducibility ---
    prepare_experiment(config)
    save_dataset_metadata(config, features, labels, groups, fold_ids)

    # --- Data / mode checks ---
    check_training_data(features, labels, config)

    # --- Hyperparameter tuning ---
    # Optional Optuna stage: search -> best params -> normal full training.
    if bool(config.tuning.enabled):
        best_params, study = run_tuning(
            config,
            features,
            labels,
            groups=groups,
            fold_ids=fold_ids,
        )
        apply_best_params(config, best_params)
        save_config_snapshot(config)

        TelegramNotifier(config).send(
            f"{config.general.experiment_name} tuning finished | "
            f"best={study.best_value:.6f} | params={best_params}"
        )

    # --- Base training ---
    train_ids = (_load_array(config.paths.path_to_train_ids) if config.paths.path_to_train_ids else None)
    fold_results = train(config, features, labels, groups=groups, fold_ids=fold_ids, ids=train_ids)

    # --- Optional self-training stage ---
    self_training_manifest = None
    if bool(config.strategies.self_training.enabled):
        self_training_manifest = run_self_training(
            config,
            features,
            labels,
            base_checkpoint_root=config.paths.path_to_fold_checkpoints,
        )

    if Path(config.paths.path_to_test_features).exists():
        inference(config)

    # --- Model conversion / export ---
    # Export the final self-training model when self-training produced one;
    # otherwise export the first base fold.
    if bool(config.conversion.enabled):
        checkpoint = None
        if self_training_manifest and self_training_manifest.get("final_checkpoint_root"):
            root = self_training_manifest["final_checkpoint_root"]
            if bool(config.split.all_data_train):
                checkpoint = f"{root}/all_data/best.pt"
            else:
                first_fold = int(config.split.folds_to_train[0])
                checkpoint = f"{root}/fold_{first_fold}/best.pt"
        elif fold_results:
            checkpoint = fold_results[0]["best_checkpoint"]
        if checkpoint:
            exported = export_model(config, checkpoint)
            if config.logging.prints:
                print("Exported:", exported)


def main():
    if config.general.mode == "train":
        fit(config)
    elif config.general.mode == "inference":
        ensure_directories(config)
        inference(config)
    else:
        raise ValueError("config.general.mode must be 'train' or 'inference'")


if __name__ == "__main__":
    main()
