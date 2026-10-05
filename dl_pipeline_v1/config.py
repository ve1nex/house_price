from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent

config = {
    "general": {
        "experiment_name": "mlp_fix_best",
        "seed": 1027309,
        "task": "regression",
        "mode": "inference",
        "overwrite_experiment": False,
    },
    "paths": {
        "path_to_test_features": str(PROJECT_ROOT / "data/test_features.npy"),
        "path_to_checkpoints": str(
            PROJECT_ROOT / "checkpoints/${general.experiment_name}"
        ),
        "path_to_fold_checkpoints": "${paths.path_to_checkpoints}/folds",
        "path_to_oof": "${paths.path_to_checkpoints}/oof_predictions.csv",
        "path_to_oof_raw": "${paths.path_to_checkpoints}/oof_raw_outputs.npz",
        "path_to_predictions": "${paths.path_to_checkpoints}/predictions.csv",
        "path_to_plots": "${paths.path_to_checkpoints}/plots",
        "path_to_config_snapshot": "${paths.path_to_checkpoints}/config.yaml",
        "path_to_metadata": "${paths.path_to_checkpoints}/metadata.json",
        "path_to_environment": "${paths.path_to_checkpoints}/environment.json",
        "path_to_logs": str(PROJECT_ROOT / "logs"),
        "path_to_optuna_root": str(PROJECT_ROOT / "optuna"),
        "path_to_optuna_db": "${paths.path_to_optuna_root}/${general.experiment_name}.db",
        "path_to_tuning": "${paths.path_to_checkpoints}/tuning",
        "path_to_tuning_trials": "${paths.path_to_tuning}/trials.csv",
        "path_to_best_params": "${paths.path_to_tuning}/best_params.yaml",
        "path_to_results_csv": "${paths.path_to_logs}/results.csv",
        "path_to_results_txt": "${paths.path_to_logs}/results.txt",
        "path_to_train_dataset": str(
            PROJECT_ROOT / "../classic_ml_pipeline_v8/data/train.csv"
        ),
        "path_to_test_dataset": str(
            PROJECT_ROOT / "../classic_ml_pipeline_v8/data/test.csv"
        ),
    },
    "reproducibility": {
        "save_config_snapshot": True,
        "save_environment": True,
        "save_data_info": True,
        "calculate_dataset_hash": False,
        "deterministic": True,
    },
    "data": {
        "target": "SalePrice",
        "id_column": "Id",
        "id_namespace": "house_prices_id",
        "target_transform": "log1p",
        "drop_train_ids": [524, 1299, 1397],
        "drop_columns": ["GarageArea", "GarageYrBlt", "Exterior2nd", "GarageCond"],
    },
    "split": {
        "strategy": "KFold",
        "n_splits": 5,
        "folds_to_train": [0, 1, 2, 3, 4],
        "folds_to_inference": [0, 1, 2, 3, 4],
        "shuffle": True,
    },
    "training": {
        "num_epochs": 300,
        "device": "auto",
        "save_best": True,
        "save_last": True,
    },
    "dataloader_params": {"batch_size": 16, "shuffle": True, "drop_last": False},
    "regularization": {
        "enabled": True,
        "dropout": {"enabled": True, "p": 0.05641011715961437},
        "weight_decay": {"enabled": True, "value": 1.0899202432093064e-05},
    },
    "optimization": {
        "enabled": True,
        "training_control": {
            "early_stopping": {"enabled": True, "patience": 25},
            "gradient_clipping": {"enabled": True, "max_norm": 1.0},
            "scheduler": {"enabled": True},
        },
    },
    "tuning": {
        "enabled": False,
        "n_trials": 300,
        "direction": "auto",
        "sampler": "tpe",
        "study_name": "${general.experiment_name}_tuning",
        "resume_study": False,
        "folds_to_use": [0, 1, 2],
        "pruning": {"enabled": True, "n_startup_trials": 10, "n_warmup_steps": 10},
        "search_space": {
            "learning_rate": {
                "path": "optimizer.params.lr",
                "type": "float",
                "low": 1e-05,
                "high": 0.005,
                "log": True,
            },
            "batch_size": {
                "path": "dataloader_params.batch_size",
                "type": "categorical",
                "choices": [16, 32, 64, 128],
            },
            "hidden_dim_1": {
                "path": "model.params.hidden_dims.0",
                "type": "categorical",
                "choices": [16, 32, 64, 128, 256, 512],
            },
            "hidden_dim_2": {
                "path": "model.params.hidden_dims.1",
                "type": "categorical",
                "choices": [8, 16, 32, 64, 128, 256],
            },
            "hidden_dim_3": {
                "path": "model.params.hidden_dims.2",
                "type": "categorical",
                "choices": [8, 16, 32, 64, 128],
            },
            "activation": {
                "path": "model.params.activation",
                "type": "categorical",
                "choices": ["ReLU", "GELU", "SiLU"],
            },
            "dropout": {
                "path": "regularization.dropout.p",
                "type": "float",
                "low": 0.0,
                "high": 0.5,
            },
            "weight_decay": {
                "path": "regularization.weight_decay.value",
                "type": "float",
                "low": 1e-08,
                "high": 0.01,
                "log": True,
            },
            "optimizer": {
                "path": "optimizer.name",
                "type": "categorical",
                "choices": ["Adam", "AdamW"],
            },
            "scheduler_enabled": {
                "path": "optimization.training_control.scheduler.enabled",
                "type": "categorical",
                "choices": [False, True],
            },
            "scheduler_T_max": {
                "path": "scheduler.params.T_max",
                "type": "int",
                "low": 20,
                "high": 150,
                "step": 10,
            },
            "scheduler_eta_min": {
                "path": "scheduler.params.eta_min",
                "type": "float",
                "low": 1e-07,
                "high": 0.0001,
                "log": True,
            },
        },
    },
    "model": {
        "name": "MLP",
        "input_shape": [293],
        "params": {"hidden_dims": [64, 32, 128], "activation": "SiLU"},
    },
    "optimizer": {
        "name": "Adam",
        "params": {"lr": 0.0010697325967296643, "weight_decay": 0.0},
    },
    "scheduler": {
        "name": "CosineAnnealingLR",
        "interval": "epoch",
        "params": {"T_max": 110, "eta_min": 5.289610681565301e-06},
    },
    "loss": {"name": "MSELoss", "params": {}},
    "metric": {
        "name": "root_mean_squared_error",
        "direction": "minimize",
        "params": {},
    },
    "visualization": {
        "save_training_curves": True,
        "save_validation_plot": True,
        "save_cv_scores": True,
    },
    "logging": {"prints": True, "txt_file": True, "csv_file": True},
}

config = OmegaConf.create(config)
