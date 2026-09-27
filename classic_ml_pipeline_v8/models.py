from omegaconf import OmegaConf
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from importlib import import_module


MODELS = {
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,

    "XGBClassifier": ("xgboost", "XGBClassifier"),
    "LGBMClassifier": ("lightgbm", "LGBMClassifier"),
    "CatBoostClassifier": ("catboost", "CatBoostClassifier"),

    "LinearRegression": LinearRegression,
    "KNeighborsRegressor": KNeighborsRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,

    "XGBRegressor": ("xgboost", "XGBRegressor"),
    "LGBMRegressor": ("lightgbm", "LGBMRegressor"),
    "CatBoostRegressor": ("catboost", "CatBoostRegressor"),
}

CLASSIFICATION_MODELS = {
    "LogisticRegression",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
}

REGRESSION_MODELS = {
    "LinearRegression",
    "KNeighborsRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "XGBRegressor",
    "LGBMRegressor",
    "CatBoostRegressor",
}


def _build_linear_regression(params):
    """Select an sklearn regressor using the one public LinearRegression config."""
    mode = str(params["regularization"]).lower()
    allowed = {"none", "ridge", "lasso", "elasticnet"}
    if mode not in allowed:
        raise ValueError(f"models.LinearRegression.regularization must be one of {sorted(allowed)}")

    common = {
        "fit_intercept": bool(params["fit_intercept"]),
        "positive": bool(params["positive"]),
    }
    if mode == "none":
        return LinearRegression(**common, n_jobs=int(params["n_jobs"]))

    alpha = float(params["alpha"])
    if alpha <= 0:
        raise ValueError("models.LinearRegression.alpha must be > 0 for regularized modes; use 'none' for OLS")
    max_iter = int(params["max_iter"])
    tol = float(params["tol"])
    if max_iter < 1 or tol <= 0:
        raise ValueError("models.LinearRegression.max_iter and tol must be positive")
    iterative = {**common, "alpha": alpha, "max_iter": max_iter, "tol": tol,
                 "random_state": params["random_state"]}

    if mode == "ridge":
        solver = str(params["solver"])
        if common["positive"] and solver not in {"auto", "lbfgs"}:
            raise ValueError("Ridge with positive=True requires solver='auto' or 'lbfgs'")
        return Ridge(**iterative, solver=solver)

    selection = str(params["selection"])
    if selection not in {"cyclic", "random"}:
        raise ValueError("models.LinearRegression.selection must be 'cyclic' or 'random'")
    if mode == "lasso":
        return Lasso(**iterative, selection=selection)

    ratio = float(params["l1_ratio"])
    if not 0 <= ratio <= 1:
        raise ValueError("models.LinearRegression.l1_ratio must be in [0, 1]")
    return ElasticNet(**iterative, l1_ratio=ratio, selection=selection)


def validate_model_for_task(name, config) -> None:
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS)}")

    if config.general.task == "classification" and name not in CLASSIFICATION_MODELS:
        raise ValueError(f"{name} is not configured as a classification model")

    if config.general.task == "regression" and name not in REGRESSION_MODELS:
        raise ValueError(f"{name} is not configured as a regression model")


def get_model_by_name(name, config):
    """Create one known sklearn model by name from the shared config."""
    validate_model_for_task(name, config)
    params = OmegaConf.to_container(config.models[name], resolve=True)
    if name == "LogisticRegression":
        ratio, solver, c = params["l1_ratio"], params["solver"], params["C"]
        if not 0 <= ratio <= 1 or (ratio > 0 and solver != "saga" and not (ratio == 1 and solver == "liblinear")):
            raise ValueError("LogisticRegression: L1/ElasticNet requires saga (L1 also supports liblinear)")
        if c != float("inf") and c <= 0:
            raise ValueError("LogisticRegression.C must be positive or infinity")
    if name in {"XGBClassifier", "LGBMClassifier"}:
        count = config.general.num_classes
        if count is None:
            raise ValueError("Set general.num_classes from training labels before creating a boosting classifier")
        params["objective"] = ("multi:softprob" if int(count) > 2 else "binary:logistic") if name == "XGBClassifier" else ("multiclass" if int(count) > 2 else "binary")
        if name == "XGBClassifier" and int(count) > 2:
            params["num_class"] = int(count)
            params.pop("scale_pos_weight", None)  # binary-only parameter
    if name == "CatBoostClassifier":
        count = config.general.num_classes
        if count is None:
            raise ValueError("Set general.num_classes before building CatBoostClassifier")
        params["loss_function"] = "MultiClass" if int(count) > 2 else "Logloss"
    if bool(config.optimization.enabled) and bool(config.optimization.speed.parallel_cpu.enabled) and "n_jobs" in params:
        params["n_jobs"] = int(config.optimization.speed.parallel_cpu.n_jobs)
    if bool(config.optimization.enabled) and bool(config.optimization.speed.parallel_cpu.enabled) and "thread_count" in params:
        params["thread_count"] = int(config.optimization.speed.parallel_cpu.n_jobs)
    if bool(config.optimization.enabled) and bool(config.optimization.speed.histogram_boosting.enabled):
        if not name.startswith("XGB"):
            raise ValueError("histogram_boosting flag applies only to XGB estimators; LightGBM uses hist inherently")
        params["tree_method"] = "hist"
    if name == "LinearRegression":
        return _build_linear_regression(params)
    klass = MODELS[name]
    if isinstance(klass, tuple):
        module, symbol = klass
        try:
            klass = getattr(import_module(module), symbol)
        except ImportError as error:
            raise ImportError(f"{name} requires optional package '{module}'") from error
    return klass(**params)


def get_model(config):
    """Create the single model selected in config.model.name."""
    return get_model_by_name(str(config.model.name), config)
