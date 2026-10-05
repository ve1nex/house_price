"""Project entry-point settings; model parameters remain in the existing pipeline configs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
config = {
    "mode": "report",
    "pipeline": "final",
    "output_dir": str(ROOT / "outputs"),
    "classic_folder": "classic_ml_pipeline_v8",
    "dl_folder": "dl_pipeline_v1",
    "classic_experiment": "clean_cv_v1_classic",
    "dl_experiment": "clean_cv_v1_dl",
    "final_experiment": "clean_cv_v1_final",
    "new_experiment_prefix": "clean_cv_v2",
}
