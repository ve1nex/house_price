from pathlib import Path
import copy

import numpy as np
import optuna
from omegaconf import OmegaConf

from train import run_fold
from utils import set_seed


def _suggest_value(trial, name, spec):
    """Sample one parameter from its configured Optuna distribution."""
    param_type = str(spec["type"])

    if param_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )

    if param_type == "float":
        step = spec.get("step")
        step = None if step in (None, "null") else float(step)
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=step,
            log=bool(spec.get("log", False)),
        )

    if param_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))

    raise ValueError(f"Unknown tuning parameter type: {param_type}")


def _get_direction(config) -> str:
    """Resolve and validate the Optuna optimization direction."""
    direction = str(config.tuning.direction)
    if direction == "auto":
        direction = str(config.metric.direction)
    if direction not in {"maximize", "minimize"}:
        raise ValueError("tuning.direction must be 'auto', 'maximize' or 'minimize'")
    return direction


def _make_sampler(config):
    """Create the configured seeded Optuna sampler."""
    name = str(config.tuning.sampler).lower()
    seed = int(config.general.seed)

    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)

    raise ValueError("tuning.sampler must be 'tpe' or 'random'")


def _make_pruner(config):
    """Create the pruning policy for unpromising Optuna trials."""
    if not bool(config.tuning.pruning.enabled):
        return optuna.pruners.NopPruner()

    return optuna.pruners.MedianPruner(
        n_startup_trials=int(config.tuning.pruning.n_startup_trials),
        n_warmup_steps=int(config.tuning.pruning.n_warmup_steps),
    )


def _storage_uri(config) -> str:
    """Build the SQLite URI for the experiment tuning database."""
    path = Path(config.paths.path_to_optuna_db).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def apply_best_params(config, best_params: dict) -> None:
    """Apply Optuna's best values to their config paths."""
    for name, value in best_params.items():
        spec = config.tuning.search_space[name]
        _set_path(config, str(spec.path), value)


def _trial_config(config, trial):
    """Create a trial configuration without persistent model artifacts."""
    trial_config = copy.deepcopy(config)

    for name, spec in config.tuning.search_space.items():
        value = _suggest_value(trial, str(name), spec)
        _set_path(trial_config, str(spec.path), value)

    # Tuning should be lightweight. The final best run uses the normal settings.
    trial_config.visualization.save_training_curves = False
    trial_config.visualization.save_validation_plot = False
    trial_config.visualization.save_cv_scores = False
    trial_config.logging.prints = False
    trial_config.logging.txt_file = False
    trial_config.logging.csv_file = False

    return trial_config


def _save_tuning_results(config, study) -> None:
    """Persist trial history and best parameters for later review."""
    tuning_dir = Path(config.paths.path_to_tuning)
    tuning_dir.mkdir(parents=True, exist_ok=True)

    study.trials_dataframe().to_csv(config.paths.path_to_tuning_trials, index=False)

    params_with_paths = {}
    for name, value in study.best_params.items():
        params_with_paths[name] = {
            "path": str(config.tuning.search_space[name].path),
            "value": value,
        }

    payload = {
        "study_name": str(study.study_name),
        "direction": str(study.direction.name).lower(),
        "best_value": float(study.best_value),
        "best_params": params_with_paths,
    }
    OmegaConf.save(OmegaConf.create(payload), config.paths.path_to_best_params)


def run_tuning(config, features, labels, groups=None, fold_ids=None):
    """Run Optuna on selected DL folds and return best params + Study."""
    if len(config.tuning.search_space) == 0:
        raise ValueError("config.tuning.search_space is empty")

    folds = [int(fold) for fold in config.tuning.folds_to_use]
    if not folds:
        raise ValueError("config.tuning.folds_to_use is empty")

    def objective(trial):
        set_seed(
            int(config.general.seed),
            deterministic=bool(config.reproducibility.deterministic),
        )
        trial_config = _trial_config(config, trial)
        scores = []

        for fold_position, fold in enumerate(folds):
            result = run_fold(
                trial_config,
                features,
                labels,
                groups,
                fold_ids,
                fold,
                checkpoint_root=config.paths.path_to_tuning,
                run_tag=f"trial_{trial.number}",
                trial=trial if bool(config.tuning.pruning.enabled) else None,
                trial_step_offset=fold_position * int(trial_config.training.num_epochs),
                save_artifacts=False,
            )
            scores.append(float(result["score"]))

        return float(np.mean(scores))

    try:
        study = optuna.create_study(
            study_name=str(config.tuning.study_name),
            direction=_get_direction(config),
            sampler=_make_sampler(config),
            pruner=_make_pruner(config),
            storage=_storage_uri(config),
            load_if_exists=bool(config.tuning.resume_study),
        )
    except optuna.exceptions.DuplicatedStudyError as error:
        raise ValueError(
            "Optuna study already exists. Choose a new experiment_name or set tuning.resume_study=True after verifying data/config"
        ) from error

    study.optimize(objective, n_trials=int(config.tuning.n_trials))
    _save_tuning_results(config, study)

    if bool(config.logging.prints):
        print("\nOptuna tuning finished")
        print(f"Best score: {study.best_value:.6f}")
        print(f"Best params: {study.best_params}")

    return dict(study.best_params), study


def _set_path(config, path, value):
    """Update a nested setting, including an item of model.params.hidden_dims."""
    parts = path.split(".")
    node = config
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    node[int(parts[-1]) if parts[-1].isdigit() else parts[-1]] = value
