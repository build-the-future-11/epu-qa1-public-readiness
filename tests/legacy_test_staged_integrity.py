"""Metadata-only E5 shared-budget regression; never opens a dataset or runs it."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import run_study as study


class StagedIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        names = ("run_study.py", "epu_query_anchor.py", "epu_diagnostics.py",
                 "safe_io.py", "result_contract.py", "analyze_results.py",
                 "SCIENTIFIC_SPEC.md", "PROTOCOL.md")
        for name in names:
            (root / name).write_bytes(b"metadata-only fixture")
        return {"version": "EPU-QA1", "role": "exposed_development_only",
                "engineering_revision": "E5", "engineering_tests": "PASS",
                "files": {name: study.sha(b"metadata-only fixture") for name in names},
                "shared_attempt_directory": "successors/query_anchor_v1_1/runs",
                "predecessor_attempt_directories": ["successors/query_anchor_v1/runs"]}

    def test_e5_correct_shared_budget_metadata_is_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = self.fixture(root)
            (root / "FREEZE.json").write_text(json.dumps(freeze), encoding="utf-8")
            self.assertEqual(study.check_freeze(root)[1], freeze)

    def test_e5_missing_or_altered_shared_budget_metadata_is_rejected(self):
        changes = [("shared_attempt_directory", None),
                   ("predecessor_attempt_directories", None),
                   ("shared_attempt_directory", "successors/query_anchor_v1_e5/runs"),
                   ("predecessor_attempt_directories", []),
                   ("predecessor_attempt_directories", ["successors/query_anchor_v1_e5/runs"])]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = self.fixture(root)
            for field, value in changes:
                with self.subTest(field=field, value=value):
                    changed = copy.deepcopy(freeze)
                    if value is None:
                        changed.pop(field)
                    else:
                        changed[field] = value
                    (root / "FREEZE.json").write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "share both predecessor QA1"):
                        study.check_freeze(root)


if __name__ == "__main__":
    unittest.main()
