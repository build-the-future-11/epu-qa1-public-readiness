"""Small mocked control-flow regressions; no real dataset or resource bypass."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_study as study
from epu_query_anchor import PrototypeModel


class RunCompletionTests(unittest.TestCase):
    def test_final_retained_file_hashing_counts_toward_deadline(self):
        rows = [dict(id=f'q{i}', row_index=i, pixels=(i, 16-i), label=i) for i in range(10)]
        train = [dict(r, id=f't{i}', row_index=i+10) for i, r in enumerate(rows)]
        clock = [0.]
        original_sha = study.sha
        def slow_final_hash(raw):
            if b'"status": "STARTED"' in raw:
                clock[0] = 601.
            return original_sha(raw)
        predictions = {name: {'prediction': 0, 'elapsed_ns': 1} for name in study.METHODS}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, 'check_freeze', return_value=('fixed', {})), \
             patch.object(study.diagnostics, 'load_reference'), \
             patch.object(study.diagnostics, 'analyze', return_value={}), \
             patch.object(study.diagnostics, 'resource_admission', return_value={'status': 'PASS'}), \
             patch.object(study.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(study, 'load_digits_file', return_value=train+rows), \
             patch.object(study, 'grouped_split', return_value=(train, rows, {})), \
             patch.object(study, 'fit_class_means', return_value=PrototypeModel([[.5, -.5]], [0])), \
             patch.object(study, 'evaluate_query', return_value=predictions), \
             patch.object(study, 'sha', side_effect=slow_final_hash):
            root = Path(directory)
            code, receipt, output = study.run('dry-run', 'MOCKED_ONLY', root, 'final-hash', package=root)
            self.assertEqual(code, 2)
            self.assertEqual(receipt['status'], 'REFUSED_OR_FAILED')
            self.assertGreater(receipt['elapsed_seconds'], 600.)
            self.assertIn('finalization', receipt['error'])
            self.assertEqual(json.loads((output/'RECEIPT.json').read_text())['status'], 'REFUSED_OR_FAILED')

    def test_fresh_admission_after_prerequisites_precedes_attempt_reservation(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, 'check_freeze', return_value=('fixed', {})), \
             patch.object(study.diagnostics, 'load_reference'), \
             patch.object(study, 'verify_dry_run'), \
             patch.object(study.diagnostics, 'resource_admission', side_effect=[{'status': 'PASS'}, {'status': 'BLOCKED_RESOURCE'}]), \
             patch.object(study, 'load_digits_file', side_effect=AssertionError('data must remain closed')) as loader:
            root = Path(directory)
            code, receipt, _ = study.run('development', 'MOCKED_ONLY', root, 'fresh-admission', 'mocked-dry-receipt', package=root)
            self.assertEqual(code, 2)
            self.assertEqual(receipt['status'], 'BLOCKED_RESOURCE')
            self.assertFalse(receipt['data_payload_opened'])
            self.assertFalse((root/'runs/ATTEMPTS.jsonl').exists())
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
