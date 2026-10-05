from sklearn.ensemble import StackingRegressor, VotingRegressor

from models import get_model, get_model_by_name


def _named_estimators(config):
    """Create uniquely named base regressors from the ensemble configuration."""
    names = [str(n) for n in config.estimator_strategy.models.regression]
    if not names:
        raise ValueError("At least one base regressor is required")
    return [
        (f"model_{i}_{name.lower()}", get_model_by_name(name, config))
        for i, name in enumerate(names)
    ]


def get_estimator(config):
    """Build one regressor, an average, or the configured sklearn stacking model."""
    if not bool(config.estimator_strategy.enabled):
        return get_model(config)
    kind = str(config.estimator_strategy.type)
    estimators = _named_estimators(config)
    if kind in {"average", "weighted_average"}:
        weights = (
            list(config.estimator_strategy.weights.regression)
            if kind == "weighted_average"
            else None
        )
        if weights is not None and len(weights) != len(estimators):
            raise ValueError("One weight per base model is required")
        return VotingRegressor(estimators, weights=weights, n_jobs=1)
    if kind == "stacking":
        params = config.estimator_strategy.stacking
        return StackingRegressor(
            estimators=estimators,
            final_estimator=get_model_by_name(
                str(params.final_model.regression), config
            ),
            cv=int(params.cv),
            passthrough=bool(params.passthrough),
            n_jobs=int(params.n_jobs),
        )
    raise ValueError(
        "estimator_strategy.type must be average, weighted_average, or stacking"
    )


def get_estimator_label(config):
    """Describe the configured estimator in the local experiment log."""
    if not config.estimator_strategy.enabled:
        return str(config.model.name)
    names = ",".join(str(n) for n in config.estimator_strategy.models.regression)
    return f"{config.estimator_strategy.type}({names})"
