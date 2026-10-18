"""Tiny generated fixtures only: no real dataset, model campaign or guard bypass."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import recovery_budget as recovery
import run_study as runner


class AdmissionTests(unittest.TestCase):
    def admission(self, samples, *, advance=True):
        timer = [0.0]
        sequence = iter(samples)
        def pause(seconds):
            if advance:
                timer[0] += seconds
        return recovery.sustained_admission(lambda: next(sequence), pause, lambda: timer[0])

    def sample(self, **changes):
        return {"free_plus_speculative_bytes": 1024**3, "pageouts": 10,
                "swap_used_bytes": 100, **changes}

    def test_thirty_second_window_requires_all_31_samples(self):
        result = self.admission([self.sample()] * 31)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["samples"]), 31)
        self.assertEqual(result["samples"][-1]["elapsed_seconds"], 30)

    def test_any_low_sample_refuses_immediately(self):
        result = self.admission([self.sample(), self.sample(free_plus_speculative_bytes=1024**3-1)])
        self.assertEqual(result["status"], "BLOCKED_RESOURCE")
        self.assertEqual(len(result["samples"]), 2)

    def test_pageout_change_and_swap_increase_refuse(self):
        for changes in [{"pageouts": 11}, {"pageouts": 9}, {"swap_used_bytes": 101}]:
            with self.subTest(changes=changes):
                self.assertEqual(self.admission([self.sample(), self.sample(**changes)])["status"], "BLOCKED_RESOURCE")

    def test_nonincreasing_swap_is_allowed(self):
        self.assertEqual(self.admission([self.sample(swap_used_bytes=100-i) for i in range(31)])["status"], "PASS")

    def test_measurement_failures_and_invalid_values_refuse(self):
        for samples in [[], [self.sample(pageouts=True)], [self.sample(swap_used_bytes=-1)], [{}]]:
            with self.subTest(samples=samples):
                result = self.admission(samples)
                self.assertEqual(result["status"], "BLOCKED_RESOURCE")
                self.assertIn("error", result)

    def test_mock_clock_cannot_skip_observation_duration(self):
        result = self.admission([self.sample()] * 31, advance=False)
        self.assertEqual(result["status"], "BLOCKED_RESOURCE")
        self.assertIn("interval", result["error"])


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.canonical = self.root / "RESEARCH_STATE.json"
        self.stack.enter_context(patch.object(recovery, "RECOVERY_BINDING",
            {**recovery.RECOVERY_BINDING, "canonical_state": str(self.canonical)}))
        self.package = self.root / recovery.NATIVE_PACKAGE
        (self.package / "runs").mkdir(parents=True)
        self.freeze = {"engineering_revision": "E5-R1", "recovery": recovery.RECOVERY_BINDING,
                       "shared_attempt_directory": recovery.NATIVE_PACKAGE + "/runs",
                       "predecessor_attempt_directories": []}
        raw = json.dumps(self.freeze, sort_keys=True).encode()
        (self.package / "FREEZE.json").write_bytes(raw)
        self.freeze_sha = hashlib.sha256(raw).hexdigest()
        self.authority = {"active_study_version_id": "EPU-QA1", "versions": [{
            "version_id": "EPU-QA1", "resource_recovery": {"recovery_id": "E5-R1",
            "freeze_sha256": self.freeze_sha, "status": "ACTIVE_RESOURCE_GATED",
            "budget": {"full_replacement_attempts": 1}}}]}
        self.canonical.write_text(json.dumps(self.authority))
        self.historical = ("\n".join(json.dumps({"at": "fixture", "run_id": run_id,
            "phase": phase, "status": "DATA_OPENING_ATTEMPT",
            "freeze_sha256": recovery.PREDECESSOR_FREEZE}) for run_id, phase in
            [(recovery.DRY_RUN_ID, "dry-run"), (recovery.FAILED_RUN_ID, "development")]) + "\n").encode()
        for relative in recovery.PREDECESSOR_LEDGERS:
            path = self.root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(self.historical)
        self.stack.enter_context(patch.object(recovery, "PREDECESSOR_LEDGER_SHA", hashlib.sha256(self.historical).hexdigest()))
        self.ledger = self.package / "runs/ATTEMPTS.jsonl"

    def reservation(self, **changes):
        return {"at": "fixture", "phase": "development", "run_id": "fixture-full",
                "freeze_sha256": self.freeze_sha, "status": "DATA_OPENING_ATTEMPT",
                "recovery_version": recovery.RECOVERY_VERSION, **changes}

    def inspect(self):
        return recovery.read_recovery_attempt(self.package, self.root, self.freeze)

    def test_absent_and_empty_ledger_unspent(self):
        self.assertIsNone(self.inspect())
        self.ledger.write_bytes(b"")
        self.assertIsNone(self.inspect())

    def test_one_reservation_spends_budget(self):
        self.ledger.write_text(json.dumps(self.reservation()) + "\n")
        self.assertEqual(self.inspect()["run_id"], "fixture-full")
        with self.assertRaisesRegex(ValueError, "already spent"):
            with recovery.locked_recovery_ledger(self.package, self.root, self.freeze):
                self.fail("spent budget admitted")

    def test_partial_malformed_duplicate_or_wrong_identity_refuse(self):
        good = json.dumps(self.reservation()).encode() + b"\n"
        for raw in [b"{", good[:-1], good+good, b"\n", b"[]\n",
                    (json.dumps(self.reservation(freeze_sha256="0"*64))+"\n").encode(),
                    (json.dumps(self.reservation(phase="dry-run"))+"\n").encode()]:
            with self.subTest(raw=raw[:20]):
                self.ledger.write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_historical_change_refuses_without_repair(self):
        path = self.root / recovery.PREDECESSOR_LEDGERS[0]
        changed = self.historical + b"\n"
        path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.inspect()
        self.assertEqual(path.read_bytes(), changed)

    def test_locks_all_ledgers_and_preserves_history(self):
        with recovery.locked_recovery_ledger(self.package, self.root, self.freeze):
            with self.assertRaises(BlockingIOError):
                with recovery.locked_recovery_ledger(self.package, self.root, self.freeze):
                    self.fail("concurrent reservation admitted")
        for relative in recovery.PREDECESSOR_LEDGERS:
            self.assertEqual((self.root / relative).read_bytes(), self.historical)

    def test_symlink_ledger_refuses(self):
        target = self.root / "empty"
        target.write_bytes(b"")
        self.ledger.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.inspect()

    def test_budget_expansion_in_freeze_refuses(self):
        modified = {**self.freeze, "recovery": {**recovery.RECOVERY_BINDING, "full_data_opening_attempts": 2}}
        with self.assertRaisesRegex(ValueError, "binding or budget"):
            recovery.read_recovery_attempt(self.package, self.root, modified)

    def test_clone_package_cannot_reset_shared_budget(self):
        clone = self.root / "clone"
        clone.mkdir()
        (clone / "FREEZE.json").write_bytes((self.package / "FREEZE.json").read_bytes())
        self.ledger.write_text(json.dumps(self.reservation()) + "\n")
        self.assertEqual(recovery.read_recovery_attempt(clone, self.root, self.freeze)["run_id"], "fixture-full")
        with self.assertRaisesRegex(ValueError, "already spent"):
            with recovery.locked_recovery_ledger(clone, self.root, self.freeze):
                self.fail("clone reset budget")
        with self.assertRaisesRegex(ValueError, "exact admitted"):
            runner.run("development", self.root / "NONEXISTENT", self.root, "clone-attempt", package=clone)

    def fixture_run(self, run_id, admission, admission_side_effect=None):
        with (patch.object(runner, "check_freeze", return_value=(self.freeze_sha, self.freeze)),
              patch.object(runner.diagnostics, "load_reference", return_value=object()),
              patch.object(runner, "require_resources"),
              patch.object(runner, "verify_dry_run", return_value={"fixture": True}),
              patch.object(recovery, "sustained_admission", return_value=admission,
                           side_effect=admission_side_effect),
              patch.object(runner, "load_digits_file", side_effect=RuntimeError("generated fixture stops at data opening")) as loader):
            code, receipt, output = runner.run("development", self.root / "NONEXISTENT_DATASET",
                self.root, run_id, self.root / "FAKE_DRY_RECEIPT", package=self.package)
            return code, receipt, output, loader.call_count

    def test_blocked_admission_never_opens_data_or_spends_attempt(self):
        code, receipt, _, calls = self.fixture_run("fixture-block", {"status": "BLOCKED_RESOURCE"})
        self.assertEqual(code, 2)
        self.assertEqual(receipt["status"], "BLOCKED_RESOURCE")
        self.assertFalse(receipt["data_payload_opened"])
        self.assertEqual(calls, 0)
        self.assertIsNone(self.inspect())

    def test_postreservation_failure_preserved_and_never_retried(self):
        code, receipt, output, calls = self.fixture_run("fixture-first", {"status": "PASS"})
        self.assertEqual(code, 2)
        self.assertTrue(receipt["data_payload_opened"])
        self.assertEqual(calls, 1)
        first_bytes = (output / "RECEIPT.json").read_bytes()
        reservation = self.ledger.read_bytes()
        _, again, _, calls = self.fixture_run("fixture-second", {"status": "PASS"})
        self.assertFalse(again["data_payload_opened"])
        self.assertEqual(calls, 0)
        self.assertIn("already spent", again["error"])
        self.assertEqual(self.ledger.read_bytes(), reservation)
        self.assertEqual((output / "RECEIPT.json").read_bytes(), first_bytes)
        for relative in recovery.PREDECESSOR_LEDGERS:
            self.assertEqual((self.root / relative).read_bytes(), self.historical)

    def test_new_dry_phase_refuses(self):
        with self.assertRaisesRegex(ValueError, "no new dry"):
            runner.run("dry-run", self.root / "NONEXISTENT", self.root, "not-started", package=self.package)
        self.assertFalse((self.package / "runs/not-started").exists())

    def test_pause_during_admission_refuses_before_reservation(self):
        def pause_authority():
            self.authority["versions"][0]["resource_recovery"]["status"] = "PAUSED"
            self.canonical.write_text(json.dumps(self.authority))
            return {"status": "PASS"}
        _, receipt, _, calls = self.fixture_run("fixture-paused", None, pause_authority)
        self.assertEqual(calls, 0)
        self.assertFalse(receipt["data_payload_opened"])
        self.assertIn("authority", receipt["error"])
        self.assertIsNone(self.inspect())

    def test_inactive_changed_or_expanded_authority_refuses(self):
        for update in [{"status": "COMPLETE_PENDING_VALIDATION"}, {"freeze_sha256": "0"*64},
                       {"budget": {"full_replacement_attempts": 2}}, {"recovery_id": "R2"}]:
            with self.subTest(update=update):
                changed = json.loads(json.dumps(self.authority))
                changed["versions"][0]["resource_recovery"].update(update)
                self.canonical.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "authority"):
                    recovery.verify_authority(self.freeze_sha)

    def test_resource_measurement_failure_preserved_before_data(self):
        with (patch.object(runner, "check_freeze", return_value=(self.freeze_sha, self.freeze)),
              patch.object(runner.diagnostics, "load_reference"),
              patch.object(runner, "require_resources", side_effect=OSError("fixture measurement failure")),
              patch.object(runner, "load_digits_file") as loader):
            _, receipt, output = runner.run("development", "NONEXISTENT", self.root,
                "measurement", "FAKE_DRY", package=self.package)
            loader.assert_not_called()
            self.assertFalse(receipt["data_payload_opened"])
            self.assertIn("measurement failure", receipt["error"])
            self.assertTrue((output / "ERROR.log").is_file())
            self.assertIsNone(self.inspect())

    def test_expired_observation_window_does_not_spend(self):
        timer = [0.0]
        def expire():
            timer[0] = 601.0
            return {"status": "PASS"}
        with patch.object(runner.time, "monotonic", side_effect=lambda: timer[0]):
            _, receipt, _, calls = self.fixture_run("expired-window", None, expire)
        self.assertEqual(calls, 0)
        self.assertIn("600-second", receipt["error"])
        self.assertIsNone(self.inspect())

    def test_sync_failure_keeps_reservation_without_opening_data(self):
        with patch.object(runner.os, "fsync", side_effect=OSError("fixture sync failed")):
            _, receipt, _, calls = self.fixture_run("sync-failure", {"status": "PASS"})
        self.assertEqual(calls, 0)
        self.assertFalse(receipt["data_payload_opened"])
        self.assertIn("sync failed", receipt["error"])
        self.assertIsNotNone(self.inspect())

    def computing_fixture(self, mode):
        # Twenty generated two-dimensional images; never read a dataset file.
        rows = [dict(id=f"q{i}", row_index=i, pixels=(i, 16-i), label=i) for i in range(10)]
        train = [dict(row, id=f"t{i}", row_index=i+10) for i, row in enumerate(rows)]
        timer, calls = [0.0], [0]
        original_sha = runner.sha
        original_file_hash = runner.evidence.file_hash
        def sha(raw):
            if mode == "finalization" and b'"status": "STARTED"' in raw:
                timer[0] = 601.0
            return original_sha(raw)
        def predict(*args):
            calls[0] += 1
            if mode == "last-query" and calls[0] == 30:
                timer[0] = 601.0
            return {name: {"prediction": 0, "elapsed_ns": 1} for name in runner.METHODS}
        def file_hash(path, limit=20_000_000):
            if mode == "finalization" and Path(path).name == "STARTED.json":
                timer[0] = 601.0
            return original_file_hash(path, limit)
        with (patch.object(runner, "check_freeze", return_value=(self.freeze_sha, self.freeze)),
              patch.object(runner.diagnostics, "load_reference"),
              patch.object(runner.diagnostics, "analyze", return_value={}),
              patch.object(runner, "require_resources"),
              patch.object(runner, "verify_dry_run", return_value={"fixture": True}),
              patch.object(recovery, "sustained_admission", return_value={"status": "PASS"}),
              patch.object(runner.time, "monotonic", side_effect=lambda: timer[0]),
              patch.object(runner, "load_digits_file", return_value=train+rows),
              patch.object(runner, "grouped_split", return_value=(train, rows, {})),
              patch.object(runner, "fit_class_means", return_value=runner.PrototypeModel([[.5, -.5]], [0])),
              patch.object(runner, "evaluate_query", side_effect=predict),
              patch.object(runner, "sha", side_effect=sha),
              patch.object(runner.evidence, "file_hash", side_effect=file_hash)):
            return runner.run("development", "GENERATED_FIXTURE_ONLY", self.root,
                "fixture-"+mode, "FAKE_DRY", package=self.package)

    def test_generated_full_control_flow_completes_all_cells(self):
        code, receipt, output = self.computing_fixture("complete")
        self.assertEqual(code, 0)
        self.assertEqual(receipt["raw_records"], 300)
        self.assertEqual(receipt["status"], "COMPLETE_EXPOSED_DEVELOPMENT")
        self.assertEqual(json.loads((output / "SUMMARY.json").read_text())["phase"], "development")

    def test_last_query_timeout_is_never_complete(self):
        code, receipt, output = self.computing_fixture("last-query")
        self.assertEqual(code, 2)
        self.assertIn("600-second", receipt["error"])
        self.assertTrue((output / "RAW.jsonl").is_file())
        self.assertFalse((output / "SUMMARY.json").exists())

    def test_final_hashing_counts_toward_deadline(self):
        code, receipt, output = self.computing_fixture("finalization")
        self.assertEqual(code, 2)
        self.assertGreater(receipt["elapsed_seconds"], 600)
        self.assertIn("finalization", receipt["error"])
        self.assertEqual(json.loads((output / "RECEIPT.json").read_text())["status"], "REFUSED_OR_FAILED")


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.package = Path(__file__).parent / "fixtures" / "predecessor"
        self.old = self.root / recovery.PREDECESSOR_PACKAGE
        self.old.mkdir(parents=True)
        original = Path(__file__).parent / "fixtures" / "predecessor"
        files = (*recovery.SCIENTIFIC_FILES, "run_study.py")
        previous = {"files": {}}
        for name in files:
            raw = (original / Path(name).name).read_bytes()
            (self.old / name).write_bytes(raw)
            previous["files"][name] = hashlib.sha256(raw).hexdigest()
        raw = json.dumps(previous).encode()
        (self.old / "FREEZE.json").write_bytes(raw)
        old_sha = hashlib.sha256(raw).hexdigest()
        self.stack.enter_context(patch.object(recovery, "PREDECESSOR_FREEZE", old_sha))
        self.dry = self.old / "runs" / recovery.DRY_RUN_ID / "RECEIPT.json"
        self.dry.parent.mkdir(parents=True)
        self.dry.write_text('{"fixture":"dry metadata only"}')
        self.stack.enter_context(patch.object(recovery, "DRY_RECEIPT_SHA", hashlib.sha256(self.dry.read_bytes()).hexdigest()))
        failed = self.old / "runs" / recovery.FAILED_RUN_ID / "RECEIPT.json"
        failed.parent.mkdir(parents=True)
        failed.write_text(json.dumps({"status": "BLOCKED_RESOURCE", "phase": "development",
            "data_payload_opened": True, "freeze_sha256": old_sha}))
        self.stack.enter_context(patch.object(recovery, "FAILED_RECEIPT_SHA", hashlib.sha256(failed.read_bytes()).hexdigest()))

    def test_scientific_source_and_execution_body_equal_predecessor(self):
        result = recovery.verify_compatibility(self.package, self.root, self.dry)
        self.assertEqual(result["status"], "PASS_IDENTICAL_SCIENTIFIC_COMPUTATION")

    def test_changed_source_or_dry_identity_refuses(self):
        self.dry.write_text('{"fixture":"changed"}')
        with self.assertRaisesRegex(ValueError, "dry receipt changed"):
            recovery.verify_compatibility(self.package, self.root, self.dry)
        (self.old / "epu_query_anchor.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            recovery.verify_compatibility(self.package, self.root, self.dry)

    def test_recovery_cannot_change_model_or_scientific_runner(self):
        candidate = self.root / "modified-candidate"
        candidate.mkdir()
        for name in (*recovery.SCIENTIFIC_FILES, "run_study.py"):
            (candidate / name).write_bytes((self.package / name).read_bytes())
        model = candidate / "epu_query_anchor.py"
        original = model.read_bytes()
        model.write_bytes(original + b"\n# unreviewed scientific edit\n")
        with self.assertRaisesRegex(ValueError, "scientific source changed"):
            recovery.verify_compatibility(candidate, self.root, self.dry)
        model.write_bytes(original)
        source = candidate / "run_study.py"
        source.write_text(source.read_text().replace("def corrupted_query(row, noise):", "def corrupted_query(row, noise):\n    raise ValueError('unreviewed edit')"))
        with self.assertRaisesRegex(ValueError, "scientific runner function changed"):
            recovery.verify_compatibility(candidate, self.root, self.dry)

    def test_existing_dry_bundle_validator_is_required_under_e5_freeze(self):
        with (patch.object(recovery, "verify_compatibility", return_value={"fixture": True}),
              patch.object(runner, "verify_completed_run", side_effect=ValueError("incomplete dry bundle")) as validate):
            with self.assertRaisesRegex(ValueError, "incomplete dry bundle"):
                runner.verify_dry_run(self.dry, "r1-freeze", self.root, self.package)
            validate.assert_called_once_with(self.dry, recovery.PREDECESSOR_FREEZE, expected_phase="dry-run")


if __name__ == "__main__":
    unittest.main()
