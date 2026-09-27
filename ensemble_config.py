"""Artifact-only ensemble configuration. Edit experiments, then run: python ensemble.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
config = {
    "general": {"experiment_name": "ensemble_v1", "overwrite_experiment": False, "seed": 0xFACED},
    "paths": {
        "classic_checkpoints": str(ROOT / "classic_ml_pipeline_v8" / "checkpoints"),
        "dl_checkpoints": str(ROOT / "dl_pipeline_v1" / "checkpoints"),
        "ensembles_root": str(ROOT / "checkpoints" / "ensembles"),
    },
    "metric": {"name": "accuracy_score", "direction": "maximize", "params": {}},
    "optimization": {"enabled": False},
    "ensemble": {
        "enabled": False,
        "preset": "boosting_dl",
        "voting": "soft",  # hard / soft when type=voting
        "weight_optimization": {"enabled": False, "method": "optuna", "n_trials": 100},
        "meta_model": {"classification": "LogisticRegression", "regression": "Ridge",
                       "params": {"classification": {"C": float("inf"), "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 1000},
                                  "regression": {"alpha": 1.0}}},
        "presets": {
            "boostings": {"type": "average", "members": [
                {"name": "xgboost", "source": "classic", "experiment": "xgb_v1"},
                {"name": "catboost", "source": "classic", "experiment": "cat_v1"}]},
            "neural_networks": {"type": "average", "members": [
                {"name": "mlp1", "source": "dl", "experiment": "mlp_v1"},
                {"name": "mlp2", "source": "dl", "experiment": "mlp_v2"}]},
            "boosting_dl": {"type": "average", "members": [
                {"name": "xgboost", "source": "classic", "experiment": "xgb_v1"},
                {"name": "mlp", "source": "dl", "experiment": "mlp_v1"}]},
            "boosting_dl_weighted": {"type": "weighted_average", "members": [
                {"name": "xgboost", "source": "classic", "experiment": "xgb_v1", "weight": 0.6},
                {"name": "mlp", "source": "dl", "experiment": "mlp_v1", "weight": 0.4}]},
            "boosting_dl_stacking": {"type": "stacking", "members": [
                {"name": "xgboost", "source": "classic", "experiment": "xgb_v1"},
                {"name": "mlp", "source": "dl", "experiment": "mlp_v1"}]},
        },
    },
    "visualization": {"save_validation_plot": True, "save_cv_scores": True,
                      "save_ensemble_diagnostics": True},
}
