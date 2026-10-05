from importlib import import_module

from omegaconf import OmegaConf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

MODELS = {
    "LinearRegression": LinearRegression,
    "KNeighborsRegressor": KNeighborsRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "XGBRegressor": ("xgboost", "XGBRegressor"),
    "LGBMRegressor": ("lightgbm", "LGBMRegressor"),
}


def _build_linear_regression(params):
    """Select an sklearn regressor using the one public LinearRegression config."""
    mode = str(params["regularization"]).lower()
    allowed = {"none", "ridge", "lasso", "elasticnet"}
    if mode not in allowed:
        raise ValueError(
            f"models.LinearRegression.regularization must be one of {sorted(allowed)}"
        )

    common = {
        "fit_intercept": bool(params["fit_intercept"]),
        "positive": bool(params["positive"]),
    }
    if mode == "none":
        return LinearRegression(**common, n_jobs=int(params["n_jobs"]))

    alpha = float(params["alpha"])
    if alpha <= 0:
        raise ValueError(
            "models.LinearRegression.alpha must be > 0 for regularized modes; use 'none' for OLS"
        )
    max_iter = int(params["max_iter"])
    tol = float(params["tol"])
    if max_iter < 1 or tol <= 0:
        raise ValueError("models.LinearRegression.max_iter and tol must be positive")
    iterative = {
        **common,
        "alpha": alpha,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": params["random_state"],
    }

    if mode == "ridge":
        solver = str(params["solver"])
        if common["positive"] and solver not in {"auto", "lbfgs"}:
            raise ValueError(
                "Ridge with positive=True requires solver='auto' or 'lbfgs'"
            )
        return Ridge(**iterative, solver=solver)

    selection = str(params["selection"])
    if selection not in {"cyclic", "random"}:
        raise ValueError(
            "models.LinearRegression.selection must be 'cyclic' or 'random'"
        )
    if mode == "lasso":
        return Lasso(**iterative, selection=selection)

    ratio = float(params["l1_ratio"])
    if not 0 <= ratio <= 1:
        raise ValueError("models.LinearRegression.l1_ratio must be in [0, 1]")
    return ElasticNet(**iterative, l1_ratio=ratio, selection=selection)


def get_model_by_name(name, config):
    """Create one of the regressors actually used in the House Prices project."""
    if name not in MODELS:
        raise ValueError(f"Unknown regressor: {name}; available: {list(MODELS)}")
    params = OmegaConf.to_container(config.models[name], resolve=True)
    if name == "LinearRegression":
        return _build_linear_regression(params)
    model_class = MODELS[name]
    if isinstance(model_class, tuple):
        module, symbol = model_class
        model_class = getattr(import_module(module), symbol)
    return model_class(**params)


def get_model(config):
    """Create the selected single regressor."""
    return get_model_by_name(str(config.model.name), config)
