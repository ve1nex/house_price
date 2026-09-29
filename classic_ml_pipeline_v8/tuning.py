from pathlib import Path
import copy

import numpy as np
import optuna
from omegaconf import OmegaConf

from tracking import log_metrics, log_summary
from train import evaluate_cv_score


def _suggest_value(trial, name, spec):
    """Create one Optuna suggestion from a small config specification."""
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
        return trial.suggest_categorical(
            name,
            list(spec["choices"]),
        )

    raise ValueError(f"Unknown tuning parameter type: {param_type}")


def _get_direction(config) -> str:
    direction = str(config.tuning.direction)

    if direction == "auto":
        direction = str(config.metric.direction)

    if direction not in {"maximize", "minimize"}:
        raise ValueError(
            "tuning.direction must be 'auto', 'maximize' or 'minimize'"
        )

    return direction


def _make_sampler(config):
    sampler_name = str(config.tuning.sampler).lower()
    seed = int(config.general.seed)

    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)

    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)

    raise ValueError("tuning.sampler must be 'tpe' or 'random'")


def _make_pruner():
    return optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1,
    )


def _storage_uri(config) -> str:
    path = Path(config.paths.path_to_optuna_db).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    return f"sqlite:///{path}"


def apply_best_params(config, best_params: dict) -> None:
    model_name = str(config.model.name)

    for param_name, value in best_params.items():
        config.models[model_name][param_name] = value


def _active_search_space(config, model_name):
    if model_name not in config.tuning.search_spaces:
        raise ValueError(
            f"No tuning search space for model '{model_name}'. "
            "Add it to config.tuning.search_spaces."
        )

    space = config.tuning.search_spaces[model_name]

    if model_name != "LinearRegression":
        return space

    mode = str(
        config.models.LinearRegression.regularization
    ).lower()

    active = {
        "none": set(),
        "ridge": {"alpha"},
        "lasso": {"alpha"},
        "elasticnet": {"alpha", "l1_ratio"},
    }

    if mode not in active:
        raise ValueError(
            f"Unknown LinearRegression regularization: {mode!r}"
        )

    missing = active[mode] - set(space)

    if missing:
        raise ValueError(
            f"LinearRegression/{mode} tuning is missing: "
            f"{sorted(missing)}"
        )

    return {
        key: spec
        for key, spec in space.items()
        if key not in {"alpha", "l1_ratio"}
        or key in active[mode]
    }


def _save_tuning_results(config, study) -> None:
    tuning_dir = Path(config.paths.path_to_tuning)
    tuning_dir.mkdir(parents=True, exist_ok=True)

    study.trials_dataframe().to_csv(
        config.paths.path_to_tuning_trials,
        index=False,
    )

    best_payload = {
        "study_name": str(study.study_name),
        "direction": str(study.direction.name).lower(),
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }

    OmegaConf.save(
        OmegaConf.create(best_payload),
        config.paths.path_to_best_params,
    )


def run_tuning(config, X, y, groups=None):
    model_name = str(config.model.name)

    search_space = _active_search_space(
        config,
        model_name,
    )

    if len(search_space) == 0 and not (
        model_name == "LinearRegression"
        and str(
            config.models.LinearRegression.regularization
        ).lower() == "none"
    ):
        raise ValueError(
            f"Search space for '{model_name}' is empty"
        )

    folds = [
        int(fold)
        for fold in config.tuning.folds_to_use
    ]

    def objective(trial):
        trial_config = copy.deepcopy(config)

        sampled_params = {}
        for param_name, spec in search_space.items():
            value = _suggest_value(
                trial,
                str(param_name),
                spec,
            )

            sampled_params[str(param_name)] = value
            trial_config.models[model_name][param_name] = value
        fold_scores = []

        for step, fold in enumerate(folds):
            fold_score = evaluate_cv_score(
                trial_config,
                X,
                y,
                groups=groups,
                folds_to_use=[fold],
            )

            fold_scores.append(float(fold_score))

            intermediate_score = float(
                np.mean(fold_scores)
            )
            trial.report(
                intermediate_score,
                step=step,
            )
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned after {step + 1}/{len(folds)} folds. "
                    f"Current CV: {intermediate_score:.6f}"
                )

        score = float(np.mean(fold_scores))
        if bool(config.tuning.log_trials_to_wandb):
            payload = {
                "tuning/trial": trial.number,
                "tuning/score": score,
            }

            for name, value in sampled_params.items():
                payload[f"tuning/params/{name}"] = value

            log_metrics(
                config,
                payload,
                step=trial.number,
            )

        return score

    try:
        study = optuna.create_study(
            study_name=str(config.tuning.study_name),
            direction=_get_direction(config),
            sampler=_make_sampler(config),
            pruner=_make_pruner(),
            storage=_storage_uri(config),
            load_if_exists=bool(
                config.tuning.resume_study
            ),
        )

    except optuna.exceptions.DuplicatedStudyError as error:
        raise ValueError(
            "Optuna study already exists. "
            "Choose a new experiment_name or set "
            "tuning.resume_study=True after verifying data/config"
        ) from error

    n_trials = (
        1
        if not search_space
        else int(config.tuning.n_trials)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    _save_tuning_results(
        config,
        study,
    )

    log_summary(
        config,
        {
            "tuning_best_value": float(study.best_value),
            "tuning_best_params": dict(study.best_params),
            "tuning_trials": len(study.trials),
        },
    )

    if bool(config.logging.prints):
        complete_trials = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        )

        pruned_trials = sum(
            trial.state == optuna.trial.TrialState.PRUNED
            for trial in study.trials
        )

        print("\nOptuna tuning finished")
        print(f"Best CV: {study.best_value:.6f}")
        print(f"Best params: {study.best_params}")
        print(f"Completed trials: {complete_trials}")
        print(f"Pruned trials: {pruned_trials}")

    return dict(study.best_params), study