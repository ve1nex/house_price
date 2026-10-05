"""One entry point for archived results, CV training, and saved-model inference."""

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT, config


def report(cfg):
    """Export the saved final submission and a table of all checkpoint CV results."""
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in [cfg["classic_folder"], cfg["dl_folder"], "checkpoints/ensembles"]:
        checkpoints = (
            ROOT / folder
            if folder == "checkpoints/ensembles"
            else ROOT / folder / "checkpoints"
        )
        for metadata_path in sorted(checkpoints.glob("*/metadata.json")):
            metadata = json.loads(metadata_path.read_text())
            metrics_path = metadata_path.with_name("metrics.json")
            if metrics_path.exists():
                metadata.update(json.loads(metrics_path.read_text()))
            if metadata.get("cv_mean") is not None:
                rows.append(
                    {
                        "pipeline": folder,
                        "experiment": metadata_path.parent.name,
                        "cv_mean": metadata["cv_mean"],
                        "cv_std": metadata["cv_std"],
                        "metric": metadata.get("metric", "root_mean_squared_error"),
                    }
                )
    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No saved CV summaries found")
    results.to_csv(output / "results.csv", index=False)
    selected = ROOT / "checkpoints/ensembles" / cfg["final_experiment"]
    submission = pd.read_csv(selected / "predictions.csv")
    ids = pd.read_csv(ROOT / cfg["classic_folder"] / "data/test.csv").Id.to_numpy()
    if list(submission.columns) != ["Id", "SalePrice"] or not np.array_equal(
        submission.Id, ids
    ):
        raise ValueError("Saved final submission IDs/schema do not match test.csv")
    if not np.isfinite(submission.SalePrice).all() or (submission.SalePrice <= 0).any():
        raise ValueError("Invalid saved sale prices")
    shutil.copyfile(selected / "predictions.csv", output / "submission.csv")
    metrics = json.loads((selected / "metrics.json").read_text())
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Final CV: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"Submission: {output / 'submission.csv'}")
    return results


def _run_pipeline(cfg, pipeline, mode, experiment):
    """Use an isolated Python process for the existing Classic/DL module imports."""
    folder = ROOT / cfg[f"{pipeline}_folder"]
    subprocess.run(
        [
            sys.executable,
            str(folder / "main.py"),
            f"general.mode={mode}",
            f"general.experiment_name={experiment}",
        ],
        cwd=ROOT,
        check=True,
    )


def run(cfg, mode, pipeline):
    """Run the requested pipeline; training uses new experiment names."""
    if mode == "report":
        return report(cfg)
    classic_name, dl_name = cfg["classic_experiment"], cfg["dl_experiment"]
    if mode == "train":
        classic_name = cfg["new_experiment_prefix"] + "_classic"
        dl_name = cfg["new_experiment_prefix"] + "_dl"
    if pipeline in {"classic", "final"}:
        _run_pipeline(cfg, "classic", mode, classic_name)
    if pipeline in {"dl", "final"}:
        _run_pipeline(cfg, "dl", mode, dl_name)
    if pipeline == "final":
        from ensemble import run_ensemble
        from ensemble_config import config as ensemble_config

        ensemble_cfg = copy.deepcopy(ensemble_config)
        ensemble_cfg["general"]["experiment_name"] = (
            cfg["new_experiment_prefix"] + "_final"
        )
        members = ensemble_cfg["ensemble"]["presets"]["classic_mlp_average"]["members"]
        members[0]["experiment"], members[1]["experiment"] = classic_name, dl_name
        directory, metrics = run_ensemble(ensemble_cfg)
        output = Path(cfg["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / "predictions.csv", output / "submission.csv")
        print(f"Final CV: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    return None


def main():
    """Parse the common project CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["report", "train", "inference"], default=config["mode"]
    )
    parser.add_argument(
        "--pipeline", choices=["classic", "dl", "final"], default=config["pipeline"]
    )
    args = parser.parse_args()
    run(config, args.mode, args.pipeline)


if __name__ == "__main__":
    main()
