"""Check regression artifact alignment and log-space averaging without fitting models."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensemble import run_ensemble
from ensemble_config import config


class EnsembleTests(unittest.TestCase):
    """Protect sample identity, fold alignment, and saved prediction scale."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = deepcopy(config)
        self.cfg["paths"].update(
            classic_checkpoints=str(self.root / "classic"),
            dl_checkpoints=str(self.root / "dl"),
            ensembles_root=str(self.root / "out"),
        )
        self.cfg["general"]["experiment_name"] = "verified_average"
        self.cfg["visualization"] = dict.fromkeys(self.cfg["visualization"], False)
        self.ids = np.arange(6) + 1
        for source, name, offset in [
            ("classic", "ensemble_sklearn_stacking", 0.1),
            ("dl", "mlp_fix_best", -0.1),
        ]:
            directory = self.root / source / name
            directory.mkdir(parents=True)
            target = np.arange(6, dtype=float) + 10
            frame = pd.DataFrame(
                {
                    "id": self.ids,
                    "target": target,
                    "prediction": target + offset,
                    "fold": np.arange(6) % 3,
                }
            )
            if source == "dl":
                frame = frame.sample(frac=1, random_state=2)
            frame.to_csv(directory / "oof_predictions.csv", index=False)
            pd.DataFrame(
                {"Id": [8, 9], "SalePrice": np.expm1(np.array([11.0, 12.0]) + offset)}
            ).to_csv(directory / "predictions.csv", index=False)
            (directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "task": "regression",
                        "n_train": 6,
                        "oof_complete": True,
                        "id_namespace": "house_prices_id",
                    }
                )
            )

    def tearDown(self):
        self.temp.cleanup()

    def _dl_path(self):
        """Return the shuffled fixture OOF file."""
        return self.root / "dl/mlp_fix_best/oof_predictions.csv"

    def test_average_aligns_ids_and_preserves_scale(self):
        directory, metrics = run_ensemble(self.cfg)
        oof = pd.read_csv(directory / "oof_predictions.csv")
        test = pd.read_csv(directory / "predictions.csv")
        np.testing.assert_array_equal(oof.id, self.ids)
        np.testing.assert_allclose(oof.prediction, oof.target, atol=1e-12)
        np.testing.assert_allclose(test.SalePrice, np.expm1([11.0, 12.0]))
        self.assertLess(metrics["cv_mean"], 1e-12)

    def test_duplicate_ids_rejected(self):
        path = self._dl_path()
        frame = pd.read_csv(path)
        frame.loc[1, "id"] = frame.loc[0, "id"]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            run_ensemble(self.cfg)

    def test_target_mismatch_rejected(self):
        path = self._dl_path()
        frame = pd.read_csv(path)
        frame.loc[0, "target"] += 2
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "target differs"):
            run_ensemble(self.cfg)

    def test_fold_mismatch_rejected(self):
        path = self._dl_path()
        frame = pd.read_csv(path)
        frame.loc[0, "fold"] = 99
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "fold assignments differ"):
            run_ensemble(self.cfg)

    def test_incomplete_oof_rejected(self):
        path = self._dl_path().with_name("metadata.json")
        metadata = json.loads(path.read_text())
        metadata["oof_complete"] = False
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "complete regression OOF"):
            run_ensemble(self.cfg)

    def test_existing_experiment_preserved(self):
        run_ensemble(self.cfg)
        with self.assertRaises(FileExistsError):
            run_ensemble(self.cfg)


if __name__ == "__main__":
    unittest.main()
