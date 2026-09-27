from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent

config = {
    # --- General ---
    "general": {
        "experiment_name": "dl_baseline_v2",
        "seed": 0xFACED,
        "task": "classification",          # classification / regression
        "mode": "train",                   # train / inference
        "num_classes": 3,
        "overwrite_experiment": False,
    },

    # --- Paths ---
    "paths": {
        "path_to_train_features": str(PROJECT_ROOT / "data" / "train_features.npy"),
        "path_to_train_labels": str(PROJECT_ROOT / "data" / "train_labels.npy"),
        "path_to_test_features": str(PROJECT_ROOT / "data" / "test_features.npy"),
        "path_to_train_ids": None,
        "path_to_test_ids": None,

        "path_to_unlabeled_features": None,
        "path_to_groups": None,
        "path_to_folds": None,

        "path_to_checkpoints": str(PROJECT_ROOT / "checkpoints" / "${general.experiment_name}"),
        "path_to_fold_checkpoints": "${paths.path_to_checkpoints}/folds",
        "path_to_oof": "${paths.path_to_checkpoints}/oof_predictions.csv",
        "path_to_oof_raw": "${paths.path_to_checkpoints}/oof_raw_outputs.npz",
        "path_to_predictions": "${paths.path_to_checkpoints}/predictions.csv",
        "path_to_plots": "${paths.path_to_checkpoints}/plots",
        "path_to_exports": "${paths.path_to_checkpoints}/exports",
        "path_to_config_snapshot": "${paths.path_to_checkpoints}/config.yaml",
        "path_to_metadata": "${paths.path_to_checkpoints}/metadata.json",
        "path_to_environment": "${paths.path_to_checkpoints}/environment.json",
        "path_to_self_training": "${paths.path_to_checkpoints}/self_training",

        "path_to_logs": str(PROJECT_ROOT / "logs"),
        "path_to_optuna_root": str(PROJECT_ROOT / "optuna"),
        "path_to_optuna_db": "${paths.path_to_optuna_root}/${general.experiment_name}.db",
        "path_to_tuning": "${paths.path_to_checkpoints}/tuning",
        "path_to_tuning_trials": "${paths.path_to_tuning}/trials.csv",
        "path_to_best_params": "${paths.path_to_tuning}/best_params.yaml",
        "path_to_results_csv": "${paths.path_to_logs}/results.csv",
        "path_to_results_txt": "${paths.path_to_logs}/results.txt",

        "telegram_credentials": str(PROJECT_ROOT / "telegram_credits.json"),
        "tensorboard_dir": "${paths.path_to_checkpoints}/tensorboard",
    },

    # --- Reproducibility ---
    "reproducibility": {
        "save_config_snapshot": True,
        "save_environment": True,
        "save_data_info": True,
        "calculate_dataset_hash": False,
        "deterministic": True,
    },

    # --- Data ---
    "data": {
        "feature_key": None,
        "label_key": None,
        "id_namespace": "row_position",
    },

    # --- Task-specific sample preprocessing ---
    "preprocessing": {"enabled": False, "status": "placeholder; implement in preprocessing.py"},

    # --- Augmentations ---
    "augmentations": {
        "enabled": False,
        "status": "placeholder; implement in augmentations.py",
    },

    # --- Validation ---
    "split": {
        "strategy": "StratifiedKFold",     # KFold / StratifiedKFold / GroupKFold / StratifiedGroupKFold
        "n_splits": 5,
        "folds_to_train": [0, 1, 2, 3, 4],
        "folds_to_inference": [0, 1, 2, 3, 4],
        "shuffle": True,
        "already_split": False,
        "all_data_train": False,
    },

    # --- Ordinary training, without additional techniques ---
    "training": {
        "num_epochs": 30,
        "device": "auto",
        "debug": False,
        "number_of_train_debug_samples": 512,
        "number_of_val_debug_samples": 256,
        "save_best": True,
        "save_last": True,
        "resume_from_latest_checkpoint": False,
    },
    "dataloader_params": {"batch_size": 64, "shuffle": True, "drop_last": False},

    # --- Explicit regularization ---
    "regularization": {
        "enabled": False,
        "dropout": {"enabled": False, "p": 0.2},
        "weight_decay": {"enabled": False, "value": 1e-4},
    },

    # --- Extra techniques in one run; master switch overrides all children ---
    "optimization": {
        "enabled": False,
        "speed": {
            "amp": {"enabled": False},
            "compile": {"enabled": False, "backend": "inductor", "mode": "default"},
            "fused_optimizer": {"enabled": False},
            "optimized_dataloader": {"enabled": False, "num_workers": 2, "pin_memory": True,
                                     "persistent_workers": True, "prefetch_factor": 2},
        },
        "memory": {
            "gradient_accumulation": {"enabled": False, "steps": 4},
            "gradient_checkpointing": {"enabled": False, "status": "placeholder"},
        },
        "training_control": {
            "early_stopping": {"enabled": False, "patience": 7},
            "gradient_clipping": {"enabled": False, "max_norm": 1.0},
            "scheduler": {"enabled": False},
            "warmup": {"enabled": False},
            "ema": {"enabled": False, "decay": 0.999, "update_after_step": 0},
        },
        "model_compression": {
            "quantization": {"enabled": False, "status": "placeholder", "mode": "PTQ", "api": "torchao.quantization.quantize_"},
            "pruning": {"enabled": False, "status": "placeholder"},
            "knowledge_distillation": {"enabled": False, "status": "placeholder"},
        },
    },

    # --- Hyperparameter tuning / Optuna ---
    # Disabled by default. Tuning uses selected folds and then the normal pipeline
    # trains once more with the best parameters.
    "tuning": {
        "enabled": False,
        "n_trials": 20,
        "direction": "auto",            # auto -> use the primary metric direction
        "sampler": "tpe",               # tpe / random
        "study_name": "${general.experiment_name}_tuning",
        "resume_study": False,
        "trial_stage": "base_model",      # self-training and export run only after the best trial
        "folds_to_use": [0],             # usually 1 fold for fast DL search

        # Stop clearly bad trials before all epochs are finished.
        "pruning": {
            "enabled": False,
            "n_startup_trials": 5,
            "n_warmup_steps": 3,
        },

        # Generic search space. `path` points to any config value.
        # Add/remove parameters without changing tuning.py.
        "search_space": {
            "learning_rate": {
                "path": "optimizer.params.lr",
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
                "log": True,
            },
            "batch_size": {
                "path": "dataloader_params.batch_size",
                "type": "categorical",
                "choices": [32, 64, 128],
            },
        },
    },

    # --- Model ---
    # Add project-specific architectures in models.py. MLP and TransformerMLP are reusable examples.
    "model": {
        "name": "MLP",                    # MLP / TransformerMLP / custom
        "input_shape": [20],
        "params": {
            # MLP params
            "hidden_dims": [128, 64],
            "activation": "ReLU",

            # TransformerMLP params (used only when model.name=TransformerMLP)
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "head_hidden_dims": [64],
        },
    },

    # --- Model and training strategies, independent of optimization ---
    "strategies": {
        # --- Fine-tuning / transfer learning ---
        # Generic mechanism. A custom model should expose model.backbone and model.head(s).
        "finetuning": {
            "enabled": False,
            "source": "checkpoint",            # checkpoint / model_builtin
            "checkpoint_path": None,            # .pt/.pth; can be our checkpoint or a plain state_dict
            "strict_load": False,
            "freeze_backbone": True,
            "freeze_batchnorms": False,
            "unfreeze_after_epoch": 3,          # 0 -> backbone trainable from first epoch
            "backbone_lr": 1e-5,
            "head_lr": 1e-3,
        },

        # --- Metric learning ---
        # Keeps ordinary supervised loss and optionally adds an embedding loss.
        "metric_learning": {
            "enabled": False,
            "embedding_dim": 64,
            "margin": 0.3,
            "metric_loss_weight": 1.0,
            "supervised_loss_weight": 1.0,
            "normalize_embeddings": True,
            "balanced_batches": True,
            "classes_per_batch": 4,
            "samples_per_class": 4,
            "label_head": None,                 # for multi-head; None -> primary_head
        },

        # --- Hard negative mining ---
        # Used by the metric-learning triplet loss. When disabled, negatives are sampled normally.
        "hard_negative_mining": {
            "enabled": False,
            "strategy": "batch_hard",          # currently batch_hard
        },

        # --- Self-training / pseudo-label cycle ---
        # Base train -> predict unlabeled -> keep confident pseudo-labels -> retrain.
        "self_training": {
            "enabled": False,
            "confidence_threshold": 0.95,
            "max_rounds": 1,
            "min_pseudo_samples": 1,
            "save_pseudo_labels": True,
        },

        # --- Multi-head learning ---
        # If enabled, path_to_train_labels must be an .npz with arrays named like the heads below.
        # The primary head drives CV stratification / main validation score.
        "multi_head": {
            "enabled": False,
            "primary_head": "main",
            "heads": {
                "main": {
                    "task": "classification",
                    "num_outputs": 3,
                    "loss_name": "CrossEntropyLoss",
                    "loss_params": {},
                    "loss_weight": 1.0,
                    "metric_name": "accuracy_score",
                    "metric_params": {},
                    "metric_direction": "maximize",
                },
                # Example auxiliary head; remove/rename for a concrete project.
                # "aux": {
                #     "task": "regression",
                #     "num_outputs": 1,
                #     "loss_name": "MSELoss",
                #     "loss_params": {},
                #     "loss_weight": 0.2,
                #     "metric_name": "mean_squared_error",
                #     "metric_params": {},
                #     "metric_direction": "minimize",
                # },
            },
        },

    },

    # --- Optimizer ---
    "optimizer": {
        "name": "AdamW",
        "params": {
            "lr": 1e-3,
            "weight_decay": 0.0,  # overridden only by regularization.weight_decay
        },
    },

    # --- Scheduler ---
    "scheduler": {
        "name": "CosineAnnealingLR",
        "interval": "epoch",
        "params": {"T_max": 25, "eta_min": 1e-6},
        "warmup": {"epochs": 3, "start_factor": 0.1},
    },

    # --- Loss (single-head supervised mode) ---
    "loss": {
        "name": "CrossEntropyLoss",
        "params": {},
    },

    # --- Metric (single-head mode) ---
    "metric": {
        "name": "accuracy_score",
        "direction": "maximize",
        "params": {},
    },

    # --- Prediction post-processing ---
    "postprocessing": {
        "enabled": False,
        "threshold": 0.5,  # only applied to binary label output when explicitly enabled
    },

    # --- Visualization ---
    "visualization": {
        "save_training_curves": True,
        "save_validation_plot": True,
        "save_cv_scores": True,
    },

    # --- Model export / conversion ---
    "conversion": {
        "enabled": False,
        "formats": ["torch_export"],
        "opset_version": 18,
    },

    "tracking": {"wandb": False, "tensorboard": False,
                 "wandb_project_name": "DL-Pipeline", "wandb_username": None, "wandb_log_config": True},

    # --- Logging / monitoring ---
    # Local artifacts are always the source of truth. W&B is the remote dashboard.
    "logging": {
        "prints": True,
        "txt_file": True,
        "csv_file": True,
        "telegram": False,
    },
}

config = OmegaConf.create(config)
