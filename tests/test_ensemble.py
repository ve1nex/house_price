"""Artifact contract: alignment, leakage-safe meta predictions, and validation."""
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


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        n = 40
        ids = np.arange(n) + 1000
        labels = np.arange(n) % 2
        folds = (np.arange(n) // 2) % 2
        for source, p in [('classic', .75), ('dl', .65)]:
            path = self.root / source / 'one'
            path.mkdir(parents=True)
            probabilities = np.where(labels, p, 1-p)
            oof = pd.DataFrame({'id': ids, 'target': labels, 'prediction': probabilities, 'fold': folds})
            if source == 'dl':
                oof = oof.sample(frac=1, random_state=3)
            oof.to_csv(path / 'oof_predictions.csv', index=False)
            pd.DataFrame({'id': [200, 201], 'prediction': [.6, .3] if source=='classic' else [.4, .8]}).to_csv(path / 'predictions.csv', index=False)
            (path/'metadata.json').write_text(json.dumps({'oof_complete': True, 'n_train': n, 'task': 'classification', 'classes': [0,1], 'id_namespace': 'sample_id'}))
        self.cfg = {'general': {'experiment_name': 'run', 'overwrite_experiment': True, 'seed': 42},
                    'paths': {'classic_checkpoints': str(self.root/'classic'), 'dl_checkpoints': str(self.root/'dl'), 'ensembles_root': str(self.root/'out')},
                    'metric': {'name':'accuracy_score','direction':'maximize','params':{}},
                    'optimization': {'enabled': False},
                    'ensemble': {'enabled': True, 'preset': 'mix', 'voting':'soft',
                                 'weight_optimization': {'enabled':False, 'method':'optuna','n_trials':4},
                                 'meta_model': {'classification':'LogisticRegression','regression':'Ridge',
                                                'params':{'classification':{'C':1.0,'l1_ratio':0.0},'regression':{'alpha':1.0}}},
                                 'presets': {'mix': {'type':'average','members':[{'name':'a','source':'classic','experiment':'one'}, {'name':'b','source':'dl','experiment':'one'}]}}},
                    'visualization': {'save_validation_plot':True,'save_cv_scores':True,'save_ensemble_diagnostics':True}}

    def tearDown(self):
        self.temp.cleanup()

    def test_supported_ensembles(self):
        for kind in ('average','weighted_average','voting','stacking'):
            cfg = deepcopy(self.cfg)
            cfg['general']['experiment_name'] = kind
            cfg['ensemble']['presets']['mix']['type'] = kind
            if kind == 'weighted_average':
                cfg['ensemble']['presets']['mix']['members'][0]['weight'] = 7
                cfg['ensemble']['presets']['mix']['members'][1]['weight'] = 3
            if kind == 'voting':
                cfg['ensemble']['voting'] = 'hard'
            directory, metrics = run_ensemble(cfg)
            self.assertAlmostEqual(metrics['score'], 1.0)
            self.assertEqual(len(pd.read_csv(directory/'oof_predictions.csv')), 40)
            self.assertEqual(set(p.name for p in (directory/'plots').iterdir()), {'validation.png','cv_scores.png','ensemble_diagnostics.png'})

    def test_weight_optimization_is_crossfitted(self):
        cfg = deepcopy(self.cfg)
        cfg['ensemble']['presets']['mix']['type'] = 'weighted_average'
        for m in cfg['ensemble']['presets']['mix']['members']:
            m['weight'] = 1
        cfg['optimization']['enabled'] = True
        cfg['ensemble']['weight_optimization']['enabled'] = True
        directory, _ = run_ensemble(cfg)
        metadata = json.loads((directory/'metadata.json').read_text())
        self.assertTrue(metadata['weights_optimized'])
        self.assertAlmostEqual(sum(metadata['weights']), 1)
        self.assertEqual(metadata['evaluation'], 'cross-fitted meta-model/weights')

    def test_mismatch_rejected(self):
        path = self.root/'dl'/'one'/'oof_predictions.csv'
        df = pd.read_csv(path)
        df.loc[df.index[0], 'target'] = 99
        df.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError, 'target differs'):
            run_ensemble(self.cfg)

    def test_multiclass_and_regression_contracts(self):
        for task in ('classification', 'regression'):
            cfg = deepcopy(self.cfg)
            cfg['general']['experiment_name'] = task
            if task == 'regression':
                cfg['metric'].update(name='root_mean_squared_error', direction='minimize')
            for source, offset in (('classic', .0), ('dl', .03)):
                path = self.root/source/'one'
                ids = np.arange(48) + 1000
                folds = np.arange(48) % 3
                target = np.arange(48) % 3 if task == 'classification' else np.arange(48) / 10
                if task == 'classification':
                    probabilities = np.full((48, 3), .1 + offset)
                    probabilities[np.arange(48), target] = .8 - 2*offset
                    cols = {f'pred_class_{i}': probabilities[:, i] for i in range(3)}
                    test_cols = {f'pred_class_{i}': probabilities[:2, i] for i in range(3)}
                    classes = [0, 1, 2]
                else:
                    cols = {'prediction': target + offset}
                    test_cols = {'prediction': [1.2+offset, 2.1+offset]}
                    classes = None
                pd.DataFrame({'id': ids, 'target': target, **cols, 'fold': folds}).to_csv(path/'oof_predictions.csv', index=False)
                pd.DataFrame({'id': [200, 201], **test_cols}).to_csv(path/'predictions.csv', index=False)
                (path/'metadata.json').write_text(json.dumps({'oof_complete': True, 'n_train': 48,
                    'task': task, 'classes': classes, 'id_namespace': 'sample_id'}))
            directory, metrics = run_ensemble(cfg)
            self.assertTrue(np.isfinite(metrics['score']))
            expected = 'pred_class_0' if task == 'classification' else 'prediction'
            self.assertIn(expected, pd.read_csv(directory/'predictions.csv').columns)

    def test_mixed_self_training_stage_rejected(self):
        path = self.root/'dl'/'one'/'metadata.json'
        meta = json.loads(path.read_text())
        meta['test_prediction_stage'] = 'self_training'
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'self-training'):
            run_ensemble(self.cfg)


if __name__ == '__main__':
    unittest.main()
