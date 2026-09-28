from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parent


config = {
    # --- General ---
    "general": {
        "experiment_name": "baseline",
        "seed": 0xFACED,
        "task": "regression",       # classification / regression
        "num_classes": None,             # populated from training labels
        "mode": "inference",                # train / inference
        "overwrite_experiment": True,   # protect old experiment artifacts by default
    },

    # --- Paths ---
    # All paths are anchored to the project folder, so the pipeline works the same
    # regardless of the current terminal directory.
    "paths": {
        "project_root": str(PROJECT_ROOT),

        # Input data.
        "path_to_data": str(PROJECT_ROOT / "data"),
        "path_to_train_dataset": "${paths.path_to_data}/train.csv",
        "path_to_test_dataset": "${paths.path_to_data}/test.csv",

        # Per-experiment artifacts.
        "path_to_checkpoints_root": str(PROJECT_ROOT / "checkpoints"),
        "path_to_checkpoints": "${paths.path_to_checkpoints_root}/${general.experiment_name}",
        "path_to_fold_models": "${paths.path_to_checkpoints}/fold_models",
        "path_to_final_model": "${paths.path_to_checkpoints}/model.joblib",
        "path_to_oof": "${paths.path_to_checkpoints}/oof_predictions.csv",
        "path_to_predictions": "${paths.path_to_checkpoints}/predictions.csv",
        "path_to_plots": "${paths.path_to_checkpoints}/plots",
        "path_to_config_snapshot": "${paths.path_to_checkpoints}/config.yaml",
        "path_to_metadata": "${paths.path_to_checkpoints}/metadata.json",
        "path_to_environment": "${paths.path_to_checkpoints}/environment.json",

        # Global experiment history.
        "path_to_logs": str(PROJECT_ROOT / "logs"),
        "path_to_wandb": str(PROJECT_ROOT / "wandb"),
        "path_to_optuna_root": str(PROJECT_ROOT / "optuna"),
        "path_to_optuna_db": "${paths.path_to_optuna_root}/${general.experiment_name}.db",
        "path_to_tuning": "${paths.path_to_checkpoints}/tuning",
        "path_to_tuning_trials": "${paths.path_to_tuning}/trials.csv",
        "path_to_best_params": "${paths.path_to_tuning}/best_params.yaml",
        "path_to_results_csv": "${paths.path_to_logs}/results.csv",
        "path_to_results_txt": "${paths.path_to_logs}/results.txt",

        # Kept outside Git via .gitignore.
        "telegram_credentials": str(PROJECT_ROOT / "telegram_credits.json"),
    },


    # --- Reproducibility metadata ---
    "reproducibility": {
        # Save the exact resolved config used by the run.
        "save_config_snapshot": True,

        # Save Python/platform/package versions.
        "save_environment": True,

        # Save dataset shape, columns and dtypes before/after generic data stages.
        "save_data_info": True,

        # Stronger dataset identity check, but can be slow for very large files.
        "calculate_dataset_hash": False,
    },

    # --- Data ---
    "data": {
        "target": "SalePrice",
        "id_column": "Id",
        "id_namespace": "row_position",  # use a shared namespace when explicit IDs exist
        "drop_columns": [],

        # Technical utility: safely downcast numeric columns when possible.
        "reduce_memory": False,  # numeric downcasting can change precision
    },

    # --- Data checks / leakage ---
    "data_checks": {
        "enabled": True,

        # Add columns here if domain knowledge tells you they leak the target.
        # Example: a column created AFTER the event you are trying to predict.
        "known_leakage_columns": [],

        # warn / raise
        "on_suspicious_feature": "warn",
    },

    # --- Preprocessing ---
    "preprocessing": {
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "scale_numeric": False,
        "encode_categorical": True,
    },

    # --- Feature engineering ---
    # Universal stage is kept, but the base implementation is intentionally empty.
    # Put task-specific features in features.py instead of hard-coding them here.
    "feature_engineering": {
        "enabled": False,
        "status": "placeholder; implement in features.py before enabling",
    },

    # --- Validation ---
    "split": {
        # Available strategies:
        # KFold / StratifiedKFold / GroupKFold / StratifiedGroupKFold
        "strategy": "KFold",
        "n_splits": 5,
        "folds_to_train": [0, 1, 2, 3, 4],
        "shuffle": True,

        # Needed only for GroupKFold / StratifiedGroupKFold.
        # The group column is automatically excluded from model features.
        "group_column": None,
    },

    # --- Training / technical settings ---
    "training": {
        "debug": False,
        "debug_n_rows": 1000,
        "save_fold_models": False,
        "save_oof_predictions": True,
        "train_final_model": True,
        "predict_test_after_fit": True,
    },

    # --- Model ---
    # Change only `name`; parameters for common models are stored below.
    "model": {
        "name": "LinearRegression",
    },

    "models": {
        "LogisticRegression": {
            "C": float("inf"),          # explicit unregularized baseline; finite C enables L2
            "l1_ratio": 0.0,          # 0=L2, 1=L1, (0,1)=ElasticNet with saga
            "solver": "lbfgs",
            "class_weight": None,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 1e-4,
            "random_state": "${general.seed}",
        },
        "KNeighborsClassifier": {
            "n_neighbors": 15,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
            "n_jobs": 1,
        },
        "RandomForestClassifier": {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": None,
            "bootstrap": True,
            "max_samples": None,
            "criterion": "gini",
            "random_state": "${general.seed}",
            "n_jobs": 1,
        },
        "GradientBoostingClassifier": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "max_features": None,
            "loss": "log_loss",
            "n_iter_no_change": None,
            "random_state": "${general.seed}",
        },
        "LinearRegression": {
            "regularization": "none",  # none / ridge / lasso / elasticnet
            "alpha": 1.0,              # used by ridge / lasso / elasticnet
            "l1_ratio": 0.5,           # used by elasticnet
            "fit_intercept": True,
            "positive": False,
            "max_iter": 5000,          # used by ridge / lasso / elasticnet
            "tol": 1e-4,
            "random_state": "${general.seed}",
            "solver": "auto",         # ridge only; preserves the old Ridge setting
            "selection": "cyclic",    # lasso / elasticnet; random_state matters if random
            "n_jobs": 1,             # unregularized LinearRegression only
        },
        "KNeighborsRegressor": {
            "n_neighbors": 15,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
            "n_jobs": 1,
        },
        "RandomForestRegressor": {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "bootstrap": True,
            "max_samples": None,
            "criterion": "squared_error",
            "random_state": "${general.seed}",
            "n_jobs": 1,
        },
        "GradientBoostingRegressor": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "max_features": None,
            "loss": "squared_error",
            "n_iter_no_change": None,
            "random_state": "${general.seed}",
        },
        "DecisionTreeClassifier": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini",
            "ccp_alpha": 0.0,
            "class_weight": None,
            "random_state": "${general.seed}",
        },
        
        "DecisionTreeRegressor": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "squared_error",
            "ccp_alpha": 0.0,
            "random_state": "${general.seed}",
        },
        "XGBClassifier": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "gamma": 0.0,
            "min_child_weight": 1.0,
            "scale_pos_weight": 1.0,
            "tree_method": "exact",
            "objective": "binary:logistic",  # use multi:softprob when num_classes > 2, set in models.py
            "random_state": "${general.seed}",
            "n_jobs": 1,
        },

        "LGBMClassifier": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "class_weight": None,
            "min_child_samples": 20,
            "min_split_gain": 0.0,
            "boosting_type": "gbdt",
            "random_state": "${general.seed}",
            "n_jobs": 1,
            "verbosity": -1,
        },

        "CatBoostClassifier": {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 0.0,
            "random_strength": 0.0,
            "bootstrap_type": "No",
            "use_best_model": False,
            "allow_writing_files": False,
            "loss_function": "Logloss",  # switched to MultiClass for >2 classes
            "thread_count": 1,
            "random_seed": "${general.seed}",
            "verbose": False,
        },
        "XGBRegressor": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "gamma": 0.0,
            "min_child_weight": 1.0,
            "tree_method": "exact",
            "objective": "reg:squarederror",
            "random_state": "${general.seed}",
            "n_jobs": 1,
        },

        "LGBMRegressor": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "min_child_samples": 20,
            "min_split_gain": 0.0,
            "boosting_type": "gbdt",
            "random_state": "${general.seed}",
            "n_jobs": 1,
            "verbosity": -1,
        },

        "CatBoostRegressor": {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 0.0,
            "random_strength": 0.0,
            "bootstrap_type": "No",
            "use_best_model": False,
            "allow_writing_files": False,
            "loss_function": "RMSE",
            "thread_count": 1,
            "random_seed": "${general.seed}",
            "verbose": False,
        },
    },

    # --- Extra techniques within one Classic training run ---
    "optimization": {
        "enabled": False,
        "speed": {
            "parallel_cpu": {"enabled": False, "n_jobs": -1},
            "histogram_boosting": {"enabled": False},  # XGBoost exact -> hist
        },
        "training_control": {
            "boosting_early_stopping": {"enabled": False, "rounds": 30},
        },
        "prediction": {
            "threshold_tuning": {"enabled": False, "n_steps": 101},
            "calibration": {"enabled": False, "status": "placeholder"},
        },
        "features": {
            "feature_selection": {"enabled": False, "status": "placeholder"},
        },
    },

    # --- In-model sklearn meta estimators (optional strategy) ---
    # Universal ensemble block. Disabled by default, so the pipeline behaves
    # exactly like a single-model pipeline until you explicitly enable it.
    "estimator_strategy": {
        "enabled": False,

        # bagging / voting / average / weighted_average / stacking
        "type": "bagging",

        # Used by voting / average / weighted_average / stacking.
        # Keep only model names that are compatible with the current task.
        "models": {
            "classification": [
                "LogisticRegression",
                "KNeighborsClassifier",
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
                "LGBMClassifier",
                "CatBoostClassifier",
            ],
            "regression": [
                "LinearRegression",
                "KNeighborsRegressor",
                "DecisionTreeRegressor",
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "XGBRegressor",
                "LGBMRegressor",
                "CatBoostRegressor",
            ],
        },

        # Used only by weighted_average. Length must match the task's model list.
        "weights": {
            "classification": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "regression": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },

        # Classification voting only: hard / soft.
        # `average` and `weighted_average` always use soft voting for classification.
        "voting": "hard",

        # Bagging wraps the single model selected in config.model.name.
        "bagging": {
            "n_estimators": 10,
            "max_samples": 1.0,
            "max_features": 1.0,
            "bootstrap": True,
            "bootstrap_features": False,
            "n_jobs": -1,
        },

        # Stacking uses OOF predictions of the base models to train this meta-model.
        "stacking": {
            "final_model": {
                "classification": "LogisticRegression",
                "regression": "LinearRegression",
            },
            "cv": 5,
            "passthrough": False,
            "n_jobs": -1,
        },
    },

    # --- Metric ---
    # Uses sklearn.metrics by name, like the mentor's DL pipeline.
    "metric": {
        "name": "root_mean_squared_error",       # e.g. accuracy_score / f1_score / root_mean_squared_error / r2_score
        "direction": "minimize",        # maximize / minimize; also used by Optuna
        "params": {},
    },


    # --- Hyperparameter tuning / Optuna ---
    # Disabled by default. When enabled: Optuna -> CV -> best params -> normal full training.
    "tuning": {
        "enabled": False,
        "n_trials": 30,
        "direction": "auto",            # auto -> use metric.direction
        "sampler": "tpe",               # tpe / random
        "study_name": "${general.experiment_name}_tuning",
        "resume_study": False,
        "folds_to_use": "${split.folds_to_train}",
        "log_trials_to_wandb": True,

        # Search spaces are selected automatically from config.model.name.
        "search_spaces": {
        # =========================
        # Logistic Regression
        # =========================
        "LogisticRegression": {
            "C": {
                "type": "float",
                "low": 1e-4,
                "high": 1000.0,
                "log": True,
            },
        },

        # =========================
        # K-Nearest Neighbors Classifier
        # =========================
        "KNeighborsClassifier": {
            "n_neighbors": {"type": "int", "low": 3, "high": 35, "step": 2},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            "p": {"type": "categorical", "choices": [1, 2]},
        },

        # =========================
        # Decision Tree Classifier
        # =========================
        "DecisionTreeClassifier": {
            "max_depth": {
                "type": "int",
                "low": 2,
                "high": 30,
            },
            "min_samples_split": {
                "type": "int",
                "low": 2,
                "high": 30,
            },
            "min_samples_leaf": {
                "type": "int",
                "low": 1,
                "high": 20,
            },
            "max_features": {
                "type": "categorical",
                "choices": [None, "sqrt", "log2"],
            },
            "ccp_alpha": {
                "type": "float",
                "low": 0.0,
                "high": 0.05,
            },
        },

        # =========================
        # Random Forest Classifier
        # =========================
        "RandomForestClassifier": {
            "n_estimators": {
                "type": "int",
                "low": 100,
                "high": 1000,
                "step": 50,
            },
            "max_depth": {
                "type": "int",
                "low": 3,
                "high": 30,
            },
            "min_samples_split": {
                "type": "int",
                "low": 2,
                "high": 20,
            },
            "min_samples_leaf": {
                "type": "int",
                "low": 1,
                "high": 10,
            },
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", None],
            },
        },

        # =========================
        # sklearn Gradient Boosting Classifier
        # =========================
        "GradientBoostingClassifier": {
            "n_estimators": {
                "type": "int",
                "low": 50,
                "high": 500,
                "step": 25,
            },
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "log": True,
            },
            "max_depth": {
                "type": "int",
                "low": 2,
                "high": 8,
            },
            "min_samples_split": {
                "type": "int",
                "low": 2,
                "high": 20,
            },
            "min_samples_leaf": {
                "type": "int",
                "low": 1,
                "high": 10,
            },
            "subsample": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "max_features": {
                "type": "categorical",
                "choices": [None, "sqrt", "log2"],
            },
        },

        # =========================
        # XGBoost Classifier
        # =========================
        "XGBClassifier": {
            "n_estimators": {
                "type": "int",
                "low": 100,
                "high": 1200,
                "step": 50,
            },
            "learning_rate": {
                "type": "float",
                "low": 0.005,
                "high": 0.3,
                "log": True,
            },
            "max_depth": {
                "type": "int",
                "low": 2,
                "high": 12,
            },
            "min_child_weight": {
                "type": "float",
                "low": 0.5,
                "high": 20.0,
                "log": True,
            },
            "gamma": {
                "type": "float",
                "low": 0.0,
                "high": 10.0,
            },
            "subsample": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "colsample_bytree": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "reg_alpha": {
                "type": "float",
                "low": 1e-8,
                "high": 10.0,
                "log": True,
            },
            "reg_lambda": {
                "type": "float",
                "low": 1e-3,
                "high": 100.0,
                "log": True,
            },
        },

        # =========================
        # LightGBM Classifier
        # =========================
        "LGBMClassifier": {
            "n_estimators": {
                "type": "int",
                "low": 100,
                "high": 1200,
                "step": 50,
            },
            "learning_rate": {
                "type": "float",
                "low": 0.005,
                "high": 0.3,
                "log": True,
            },
            "num_leaves": {
                "type": "int",
                "low": 8,
                "high": 128,
            },
            "max_depth": {
                "type": "int",
                "low": 3,
                "high": 15,
            },
            "min_child_samples": {
                "type": "int",
                "low": 5,
                "high": 100,
            },
            "subsample": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "colsample_bytree": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "reg_alpha": {
                "type": "float",
                "low": 1e-8,
                "high": 10.0,
                "log": True,
            },
            "reg_lambda": {
                "type": "float",
                "low": 1e-8,
                "high": 10.0,
                "log": True,
            },
        },

        # =========================
        # CatBoost Classifier
        # =========================
        "CatBoostClassifier": {
            "iterations": {
                "type": "int",
                "low": 100,
                "high": 1200,
                "step": 50,
            },
            "learning_rate": {
                "type": "float",
                "low": 0.005,
                "high": 0.3,
                "log": True,
            },
            "depth": {
                "type": "int",
                "low": 4,
                "high": 10,
            },
            "l2_leaf_reg": {
                "type": "float",
                "low": 1e-3,
                "high": 100.0,
                "log": True,
            },
            "random_strength": {
                "type": "float",
                "low": 0.0,
                "high": 10.0,
            },
            "bagging_temperature": {
                "type": "float",
                "low": 0.0,
                "high": 10.0,
            },
        },

        # =========================
        # Linear Regression: tuning.py filters these according to regularization.
        # =========================
        "LinearRegression": {
            "alpha": {
                "type": "float",
                "low": 1e-4,
                "high": 1000.0,
                "log": True,
            },
            "l1_ratio": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
            },
        },

        # =========================
        # K-Nearest Neighbors Regressor
        # =========================
        "KNeighborsRegressor": {
            "n_neighbors": {"type": "int", "low": 3, "high": 35, "step": 2},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            "p": {"type": "categorical", "choices": [1, 2]},
        },

        # =========================
        # Decision Tree Regressor
        # =========================
        "DecisionTreeRegressor": {
            "max_depth": {
                "type": "int",
                "low": 2,
                "high": 30,
            },
            "min_samples_split": {
                "type": "int",
                "low": 2,
                "high": 30,
            },
            "min_samples_leaf": {
                "type": "int",
                "low": 1,
                "high": 20,
            },
            "max_features": {
                "type": "categorical",
                "choices": [None, "sqrt", "log2"],
            },
            "ccp_alpha": {
                "type": "float",
                "low": 0.0,
                "high": 0.05,
            },
        },

        # =========================
        # Random Forest Regressor
        # =========================
        "RandomForestRegressor": {
            "n_estimators": {
                "type": "int",
                "low": 100,
                "high": 1000,
                "step": 50,
            },
            "max_depth": {
                "type": "int",
                "low": 3,
                "high": 30,
            },
            "min_samples_split": {
                "type": "int",
                "low": 2,
                "high": 20,
            },
            "min_samples_leaf": {
                "type": "int",
                "low": 1,
                "high": 10,
            },
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", None],
            },
        },

        # =========================
        # sklearn Gradient Boosting Regressor
        # =========================
        "GradientBoostingRegressor": {
            "n_estimators": {
                "type": "int",
                "low": 50,
                "high": 500,
                "step": 25,
            },
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "log": True,
            },
            "max_depth": {
                "type": "int",
                "low": 2,
                "high": 8,
            },
            "min_samples_split": {
                "type": "int",
                "low": 2,
                "high": 20,
            },
            "min_samples_leaf": {
                "type": "int",
                "low": 1,
                "high": 10,
            },
            "subsample": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "max_features": {
                "type": "categorical",
                "choices": [None, "sqrt", "log2"],
            },
        },

        # =========================
        # XGBoost Regressor
        # =========================
        "XGBRegressor": {
            "n_estimators": {
                "type": "int",
                "low": 100,
                "high": 1200,
                "step": 50,
            },
            "learning_rate": {
                "type": "float",
                "low": 0.005,
                "high": 0.3,
                "log": True,
            },
            "max_depth": {
                "type": "int",
                "low": 2,
                "high": 12,
            },
            "min_child_weight": {
                "type": "float",
                "low": 0.5,
                "high": 20.0,
                "log": True,
            },
            "gamma": {
                "type": "float",
                "low": 0.0,
                "high": 10.0,
            },
            "subsample": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "colsample_bytree": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "reg_alpha": {
                "type": "float",
                "low": 1e-8,
                "high": 10.0,
                "log": True,
            },
            "reg_lambda": {
                "type": "float",
                "low": 1e-3,
                "high": 100.0,
                "log": True,
            },
        },

        # =========================
        # LightGBM Regressor
        # =========================
        "LGBMRegressor": {
            "n_estimators": {
                "type": "int",
                "low": 100,
                "high": 1200,
                "step": 50,
            },
            "learning_rate": {
                "type": "float",
                "low": 0.005,
                "high": 0.3,
                "log": True,
            },
            "num_leaves": {
                "type": "int",
                "low": 8,
                "high": 128,
            },
            "max_depth": {
                "type": "int",
                "low": 3,
                "high": 15,
            },
            "min_child_samples": {
                "type": "int",
                "low": 5,
                "high": 100,
            },
            "subsample": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "colsample_bytree": {
                "type": "float",
                "low": 0.5,
                "high": 1.0,
            },
            "reg_alpha": {
                "type": "float",
                "low": 1e-8,
                "high": 10.0,
                "log": True,
            },
            "reg_lambda": {
                "type": "float",
                "low": 1e-8,
                "high": 10.0,
                "log": True,
            },
        },

        # =========================
        # CatBoost Regressor
        # =========================
        "CatBoostRegressor": {
            "iterations": {
                "type": "int",
                "low": 100,
                "high": 1200,
                "step": 50,
            },
            "learning_rate": {
                "type": "float",
                "low": 0.005,
                "high": 0.3,
                "log": True,
            },
            "depth": {
                "type": "int",
                "low": 4,
                "high": 10,
            },
            "l2_leaf_reg": {
                "type": "float",
                "low": 1e-3,
                "high": 100.0,
                "log": True,
            },
            "random_strength": {
                "type": "float",
                "low": 0.0,
                "high": 10.0,
            },
            "bagging_temperature": {
                "type": "float",
                "low": 0.0,
                "high": 10.0,
            },
        },
    },
    },

    # --- Prediction post-processing ---
    # The stage is kept as a hook. Base pipeline does not alter predictions.
    "postprocessing": {
        "enabled": False,
        "status": "placeholder; implement in postprocessing.py before enabling",
    },

    # --- Visualization / model understanding ---
    "visualization": {
        "save_validation_plot": True,
        "save_cv_scores": True,
        "save_feature_importance": True,
        "show_shap": False,
        "shap_max_samples": 500,
    },

    # --- Local logging / notifications ---
    "logging": {
        "prints": True,
        "txt_file": True,
        "csv_file": True,

        # Turn this on after creating telegram_credits.json and running
        # `python test_telegram.py`.
        "telegram": False,
    },

    # --- Experiment tracking ---
    # Local files remain the main source of truth. W&B is an optional remote
    # dashboard for config, fold metrics, final CV score and experiment artifacts.
    "tracking": {
        "wandb": False,
        "wandb_project": "classic-ml",
        "wandb_entity": None,
        "wandb_mode": "online",      # online / offline / disabled
        "wandb_tags": ["classic-ml"],
        "log_artifacts": True,
        "log_plots": True,
        "raise_on_error": False,
    },
}


# Keep the same convenient config approach as in the mentor's DL pipeline.
config = OmegaConf.create(config)
