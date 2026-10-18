"""Pre-run regression fixtures; no scientific dataset is read."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_study as study


def records(n=100, deltas=(-100, 4, 99)):
    rows = []
    for noise, delta in zip((0., .25, .5), deltas):
        for i in range(n):
            direct = int(delta >= 0)
            anchor = (0 if i < delta else 1) if delta >= 0 else (1 if i < -delta else 0)
            for method in study.METHODS:
                rows.append(dict(query_id=f"q{i}", noise=noise, method=method, label=0,
                                 prediction=anchor if method == 'anchor8' else direct,
                                 initial_prediction=direct, elapsed_ns=1))
    return rows


class PreRunIntegrityTests(unittest.TestCase):
    def test_expired_admission_prevents_budget_and_data_access(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, 'check_freeze', return_value=('fixed', {})), \
             patch.object(study.diagnostics, 'load_reference'), \
             patch.object(study.diagnostics, 'resource_admission', return_value={'status': 'PASS'}), \
             patch.object(study.time, 'monotonic', side_effect=[0., 601., 601.]), \
             patch.object(study, 'load_digits_file') as loader:
            root = Path(directory)
            code, receipt, _ = study.run('dry-run', 'absent', root, 'late', package=root)
            self.assertEqual(code, 2)
            loader.assert_not_called()
            self.assertFalse((root/'runs/ATTEMPTS.jsonl').exists())
            self.assertIn('600-second', receipt['error'])

    def test_last_query_timeout_cannot_be_complete(self):
        # Complete small, explicitly mocked runner path; no real payload/guard.
        rows = [dict(id=f'q{i}', row_index=i, pixels=(i, 16-i), label=i) for i in range(10)]
        train = [dict(r, id=f't{i}', row_index=i+10) for i, r in enumerate(rows)]
        clock = [0.]
        model = __import__('epu_query_anchor')
        base = model.PrototypeModel([[.5, -.5]], [0])
        calls = [0]
        def predict(*args):
            calls[0] += 1
            if calls[0] == len(rows):
                clock[0] = 601.
            return {m: {'prediction': 0, 'elapsed_ns': 1} for m in study.METHODS}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, 'check_freeze', return_value=('fixed', {})), \
             patch.object(study.diagnostics, 'load_reference'), \
             patch.object(study.diagnostics, 'analyze', return_value={}), \
             patch.object(study.diagnostics, 'resource_admission', return_value={'status': 'PASS'}), \
             patch.object(study.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(study, 'load_digits_file', return_value=train+rows), \
             patch.object(study, 'grouped_split', return_value=(train, rows, {})), \
             patch.object(study, 'fit_class_means', return_value=base), \
             patch.object(study, 'evaluate_query', side_effect=predict):
            root = Path(directory)
            code, receipt, output = study.run('dry-run', 'absent', root, 'late-query', package=root)
            self.assertEqual(code, 2)
            self.assertIn('600-second', receipt['error'])
            self.assertTrue(receipt['data_payload_opened'])
            self.assertTrue((output/'RAW.jsonl').is_file())
            self.assertFalse((output/'SUMMARY.json').exists())

    def test_exact_one_percentage_point_passes(self):
        summary = study.summarize(records())
        self.assertEqual(summary['development_screen'], 'NOMINATE_FOR_FRESH_CONFIRMATION')
        self.assertEqual(summary['primary_anchor8_minus_direct'], .01)

    def test_below_and_above_threshold(self):
        self.assertEqual(study.summarize(records(deltas=(-100, 3, 99)))['development_screen'],
                         'UTILITY_SCREEN_NOT_MET')
        self.assertEqual(study.summarize(records(deltas=(-100, 5, 99)))['development_screen'],
                         'NOMINATE_FOR_FRESH_CONFIRMATION')

    def test_dry_run_never_nominates_or_closes_utility(self):
        for delta in (-2, 2):
            rows = [r for r in records(2, (delta, 0, 0)) if r['noise'] == 0.]
            summary = study.summarize(rows, phase='dry-run')
            self.assertEqual(summary['development_screen'], 'NOT_EVALUATED_DRY_RUN')

    def test_unknown_phase_refused(self):
        with self.assertRaises(ValueError):
            study.summarize(records(2, (0, 0, 0)), phase='confirmation')

    def test_label_cannot_change_across_corruptions(self):
        rows = records(2, (0, 0, 0))
        for row in rows:
            if row['noise'] == .25 and row['query_id'] == 'q0':
                row['label'] = 1
        with self.assertRaises(ValueError):
            study.summarize(rows)

    def test_malformed_result_values_refused(self):
        for field, value in [('elapsed_ns', -1), ('elapsed_ns', True), ('elapsed_ns', float('nan')),
                             ('prediction', -1), ('prediction', 10), ('label', -1),
                             ('noise', True), ('noise', .3), ('query_id', '')]:
            rows = records(2, (0, 0, 0))
            for row in rows:
                if field == 'noise' or field == 'query_id':
                    if row[field] == rows[0][field]:
                        row[field] = value
                else:
                    row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                study.summarize(rows)

    def test_duplicate_expected_axes_refused(self):
        rows = records(2, (0, 0, 0))
        for ids, noises in [(['q0', 'q1', 'q1'], [0., .25, .5]),
                            (['q0', 'q1'], [0., 0., .25, .5])]:
            with self.subTest(ids=ids, noises=noises), self.assertRaises(ValueError):
                study.summarize(rows, ids, noises)

    def test_initial_prediction_must_match_direct(self):
        rows = records(2, (0, 0, 0))
        rows[4]['initial_prediction'] = 7
        with self.assertRaises(ValueError):
            study.summarize(rows)

    def test_attempt_is_synced_before_loader(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, 'check_freeze', return_value=('fixed', {})), \
             patch.object(study.diagnostics, 'load_reference'), \
             patch.object(study.diagnostics, 'resource_admission', return_value={'status': 'PASS'}), \
             patch('os.fsync', wraps=os.fsync) as sync:
            def loader(_):
                self.assertGreater(sync.call_count, 0, 'attempt not durable before opening data')
                raise ValueError('fixture loader refusal')
            with patch.object(study, 'load_digits_file', side_effect=loader):
                _, receipt, _ = study.run('dry-run', 'absent', Path(directory), 'durable', package=Path(directory))
            self.assertEqual(receipt['error'], 'ValueError: fixture loader refusal')

    def test_sync_failure_prevents_loader(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, 'check_freeze', return_value=('fixed', {})), \
             patch.object(study.diagnostics, 'load_reference'), \
             patch.object(study.diagnostics, 'resource_admission', return_value={'status': 'PASS'}), \
             patch('os.fsync', side_effect=OSError('sync failed')), \
             patch.object(study, 'load_digits_file') as loader:
            _, receipt, _ = study.run('dry-run', 'absent', Path(directory), 'failed', package=Path(directory))
            loader.assert_not_called()
            self.assertFalse(receipt['data_payload_opened'])
            self.assertIn('sync failed', receipt['error'])

    def test_corrupted_attempt_phase_refuses_before_loader(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, 'check_freeze', return_value=('fixed', {})), \
             patch.object(study.diagnostics, 'load_reference'), \
             patch.object(study.diagnostics, 'resource_admission', return_value={'status': 'PASS'}), \
             patch.object(study, 'load_digits_file') as loader:
            root = Path(directory)
            (root/'runs').mkdir()
            (root/'runs/ATTEMPTS.jsonl').write_text(json.dumps({'phase': 'dry_run'})+'\n')
            _, receipt, _ = study.run('dry-run', 'absent', root, 'corrupt', package=root)
            loader.assert_not_called()
            self.assertFalse(receipt['data_payload_opened'])

    def test_regenerated_dry_run_keeps_phase_boundary(self):
        from test_result_contract import fixture_bundle, FIXTURE_FREEZE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_bundle(root)
            with patch.object(study, 'check_freeze', return_value=(FIXTURE_FREEZE, {})):
                result = study.regenerate(root/'RECEIPT.json', root/'regenerated')
            self.assertEqual(result['development_screen'], 'NOT_EVALUATED_DRY_RUN')


if __name__ == '__main__':
    unittest.main()
