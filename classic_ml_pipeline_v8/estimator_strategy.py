from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)

from models import get_model, get_model_by_name


def _task_section(section, config):
    """Read a classification/regression-specific config section."""
    return section[str(config.general.task)]


def _model_names(config):
    return [str(name) for name in _task_section(config.estimator_strategy.models, config)]


def _named_estimators(config):
    """Build the list of base models used by voting/averaging/stacking."""
    names = _model_names(config)
    if not names:
        raise ValueError("config.estimator_strategy.models must contain at least one model")

    # sklearn requires unique estimator names.
    return [
        (f"model_{index}_{name.lower()}", get_model_by_name(name, config))
        for index, name in enumerate(names)
    ]


def build_bagging(config):
    """Bagging: train copies of one base model on bootstrap/subsampled data."""
    base_model = get_model(config)
    params = config.estimator_strategy.bagging

    common = {
        "estimator": base_model,
        "n_estimators": int(params.n_estimators),
        "max_samples": float(params.max_samples),
        "max_features": float(params.max_features),
        "bootstrap": bool(params.bootstrap),
        "bootstrap_features": bool(params.bootstrap_features),
        "n_jobs": int(params.n_jobs),
        "random_state": int(config.general.seed),
    }

    if config.general.task == "classification":
        return BaggingClassifier(**common)
    return BaggingRegressor(**common)


def build_voting_or_average(config, ensemble_type):
    """Voting and averaging use the same sklearn family with different settings."""
    estimators = _named_estimators(config)
    weights = None

    if ensemble_type == "weighted_average":
        weights = [
            float(weight)
            for weight in _task_section(config.estimator_strategy.weights, config)
        ]
        if len(weights) != len(estimators):
            raise ValueError(
                "config.estimator_strategy.weights must have the same length as "
                "config.estimator_strategy.models"
            )

    if config.general.task == "classification":
        if ensemble_type in {"average", "weighted_average"}:
            # Probability averaging, then select the class with the largest mean probability.
            voting = "soft"
        else:
            voting = str(config.estimator_strategy.voting)
            if voting not in {"hard", "soft"}:
                raise ValueError("config.estimator_strategy.voting must be 'hard' or 'soft'")

        return VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1,
        )

    # For regression VotingRegressor is averaging; optional weights turn it into
    # weighted averaging.
    return VotingRegressor(
        estimators=estimators,
        weights=weights,
        n_jobs=-1,
    )


def build_stacking(config):
    """Stacking: base-model OOF predictions become features for a meta-model."""
    estimators = _named_estimators(config)
    params = config.estimator_strategy.stacking
    final_model_name = str(_task_section(params.final_model, config))
    final_model = get_model_by_name(final_model_name, config)

    common = {
        "estimators": estimators,
        "final_estimator": final_model,
        "cv": int(params.cv),
        "n_jobs": int(params.n_jobs),
        "passthrough": bool(params.passthrough),
    }

    if config.general.task == "classification":
        return StackingClassifier(**common)
    return StackingRegressor(**common)


def get_estimator(config):
    """Return either one model or the ensemble selected in config."""
    if not bool(config.estimator_strategy.enabled):
        return get_model(config)

    ensemble_type = str(config.estimator_strategy.type).lower()

    if ensemble_type == "bagging":
        return build_bagging(config)

    if ensemble_type in {"voting", "average", "weighted_average"}:
        return build_voting_or_average(config, ensemble_type)

    if ensemble_type == "stacking":
        return build_stacking(config)

    raise ValueError(
        "Unknown config.estimator_strategy.type. Use one of: "
        "bagging, voting, average, weighted_average, stacking"
    )


def get_estimator_label(config) -> str:
    """Human-readable model name for logs/results."""
    if not bool(config.estimator_strategy.enabled):
        return str(config.model.name)

    ensemble_type = str(config.estimator_strategy.type)
    if ensemble_type == "bagging":
        return f"bagging({config.model.name})"

    models = ",".join(_model_names(config))
    if ensemble_type == "stacking":
        final_model = _task_section(config.estimator_strategy.stacking.final_model, config)
        return f"stacking({models} -> {final_model})"
    return f"{ensemble_type}({models})"
