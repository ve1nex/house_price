"""Saved-prediction ensembles used in the House Prices experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEMBERS = [
    {"name": "classic", "source": "classic", "experiment": "ensemble_sklearn_stacking"},
    {"name": "mlp", "source": "dl", "experiment": "mlp_fix_best"},
]

config = {
    "general": {"experiment_name": "ensemble_reproduced"},
    "paths": {
        "classic_checkpoints": str(ROOT / "classic_ml_pipeline_v8/checkpoints"),
        "dl_checkpoints": str(ROOT / "dl_pipeline_v1/checkpoints"),
        "ensembles_root": str(ROOT / "checkpoints/ensembles"),
    },
    "metric": {
        "name": "root_mean_squared_error",
        "direction": "minimize",
        "params": {},
    },
    "ensemble": {
        "preset": "classic_mlp_average",
        "target_transform": "log1p",
        "meta_model": {"name": "Ridge", "params": {"alpha": 1.0}},
        "presets": {
            "classic_mlp_average": {"type": "average", "members": MEMBERS},
            "classic_mlp_stacking": {"type": "stacking", "members": MEMBERS},
        },
    },
    "visualization": {
        "save_validation_plot": True,
        "save_cv_scores": True,
        "save_ensemble_diagnostics": True,
    },
}
