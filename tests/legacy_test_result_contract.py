"""Constructed integrity fixtures, never dataset or accuracy evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import result_contract as evidence
import run_study as study
from epu_query_anchor import PrototypeModel, unit_center

FIXTURE_FREEZE = "e" * 64


def resource_fixture():
    return {"status": "PASS", "floor_bytes": 536870912,
            "samples": [{"free_plus_speculative_bytes": 600_000_000, "pageouts": 0, "swap_used_bytes": 0} for _ in range(2)]}


def records(count=32, noises=(0.,)):
    for noise in noises:
        for index in range(count):
            label = index % 10
            for method in study.METHODS:
                yield {"query_id": f"digits:{index}", "noise": noise, "method": method,
                       "label": label, "prediction": (label + 1) % 10 if method == "anchor8" and index % 11 == 0 else label,
                       "initial_prediction": label, "elapsed_ns": 17 + index,
                       "ignored_diagnostic_payload": "x" * 200}


def fixture_bundle(root):
    """Complete-shaped handcrafted bundle for verification tests only."""
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in records())
    (root / "RAW.jsonl").write_bytes(raw)
    summary = study.summarize(records(), [f"digits:{i}" for i in range(32)], [0.], phase="dry-run")
    study.dump_new(root / "SUMMARY.json", summary)
    study.write_tables(root, summary)
    split = {"role": "historically_exposed_development", "query_ids": [f"digits:{i}" for i in range(32)],
             "training_ids": [f"digits:{i}" for i in range(450, 1797)],
             "all_query_aliases": {f"digits:{i}": [f"digits:{i}"] for i in range(450)}}
    study.dump_new(root / "SPLIT.json", split)
    model = PrototypeModel([unit_center([int(i == j) for j in range(64)]) for i in range(10)], list(range(10)))
    model.save(root / "MODEL.json")
    study.dump_new(root / "BOUNDS.json", model.bounds())
    # This verifier hashes diagnostics; it does not independently recompute traces.
    for name in ("NATIVE_DIAGNOSTICS.json", "ANCHOR_TRACES.json", "STARTED.json"):
        study.dump_new(root / name, {"constructed_integrity_fixture": True})
    outputs = {name: evidence.file_hash(root / name) for name in evidence.OUTPUT_NAMES}
    receipt = {"version": "EPU-QA1", "status": "COMPLETE_EXPOSED_DEVELOPMENT", "phase": "dry-run",
               "scientific_disposition": "NOT_CONFIRMATORY", "data_payload_opened": True,
               "protected_outcomes_opened": False, "freeze_sha256": FIXTURE_FREEZE,
               "data_sha256": study.DATA_SHA256, "native_source_sha256": study.diagnostics.REFERENCE_SHA256,
               "query_ids": split["query_ids"], "noises": [0.], "raw_records": 320,
               "query_groups": 32, "training_rows": 1347, "outputs": outputs,
               "elapsed_seconds": 1., "max_rss_bytes": 1,
               "resource_checks": [resource_fixture(), resource_fixture()],
               "retained_files": {**outputs, "STARTED.json": evidence.file_hash(root / "STARTED.json")}}
    study.dump_new(root / "RECEIPT.json", receipt)
    return receipt


def rewrite(root, receipt, name, value):
    raw = (json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n").encode()
    (root / name).write_bytes(raw)
    receipt["outputs"][name] = receipt["retained_files"][name] = study.sha(raw)
    (root / "RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")


class SummaryTests(unittest.TestCase):
    def test_single_pass_and_order_independence(self):
        original = list(records(5, (0., .25, .5)))
        expected = evidence.summarize(iter(original))
        random.Random(39).shuffle(original)
        self.assertEqual(evidence.summarize(iter(original)), expected)

    def test_bad_types_domains_and_timing_refuse(self):
        for key, values in {"elapsed_ns": [-1, .5, True, 600_000_000_001],
                            "prediction": [-1, 10, True], "label": [-1, 10, True],
                            "noise": [False, float("nan"), .1], "initial_prediction": [True, 20]}.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    rows = list(records(1)); rows[0][key] = value
                    with self.assertRaises(ValueError): evidence.summarize(rows)

    def test_label_stable_across_conditions(self):
        rows = list(records(1, (0., .25)))
        for row in rows:
            if row["noise"] == .25: row["label"] = 8
        with self.assertRaisesRegex(ValueError, "label changes"): evidence.summarize(rows)

    def test_initial_and_repeated_direct_invariants(self):
        for key in ("initial_prediction", "prediction"):
            rows = list(records(1))
            next(row for row in rows if row["method"] == "direct_repeat8")[key] = 8
            with self.assertRaises(ValueError): evidence.summarize(rows)

    def test_duplicate_empty_extra_and_missing_cells_refuse(self):
        rows = list(records(2))
        for invalid in ([], rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError): evidence.summarize(invalid, ["digits:0", "digits:1"], [0.])
        for queries, noises in ((["digits:0", "digits:0"], [0.]), (["digits:0"], [0., 0.]), (["digits:0"], [False])):
            with self.assertRaises(ValueError): evidence.summarize(rows, queries, noises)

    def test_compact_accumulator_does_not_retain_raw_rows(self):
        accumulator = evidence.SummaryAccumulator()
        row = next(records(1))
        accumulator.add(row)
        row["prediction"] = 9
        self.assertEqual(accumulator.cells["digits:0", 0., "direct"], (0, 0))
        self.assertNotIn("ignored_diagnostic_payload", repr(accumulator.__dict__))


class EvidenceTests(unittest.TestCase):
    def test_mocked_runner_artifacts_pass_regeneration_contract(self):
        # Only this test's generated fixture is loaded. No scientific dataset run.
        reference = study.diagnostics.load_reference(Path(os.environ.get(
            "EPU_REFERENCE_ROOT", str(Path(__file__).resolve().parents[2]))))
        rows = [{"id": f"digits:{i}", "row_index": i, "label": i % 10,
                 "pixels": tuple([(i // (17**j)) % 17 for j in range(4)] +
                                 [16 if j == i % 10 else 0 for j in range(60)])} for i in range(1797)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(study, "check_freeze", return_value=(FIXTURE_FREEZE, {})), \
                 patch.object(study.diagnostics, "load_reference", return_value=reference), \
                 patch.object(study.diagnostics, "resource_admission", side_effect=lambda: resource_fixture()), \
                 patch.object(study, "load_digits_file", return_value=rows):
                code, receipt, output = study.run("dry-run", "FIXTURE_ONLY", root, "fixture-integration", package=root)
                self.assertEqual(code, 0, receipt.get("error"))
                self.assertEqual(receipt["raw_records"], 320)
                study.verify_dry_run(output / "RECEIPT.json", FIXTURE_FREEZE)
                study.regenerate(output / "RECEIPT.json", root / "regenerated")

    def test_resource_and_runtime_contradictions_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); receipt = fixture_bundle(root)
            failed = resource_fixture(); failed["samples"][1]["pageouts"] = 1
            for key, value in (("resource_checks", []), ("resource_checks", [failed, resource_fixture()]),
                               ("elapsed_seconds", 601.), ("elapsed_seconds", True), ("max_rss_bytes", 0)):
                with self.subTest(key=key, value=value):
                    (root / "RECEIPT.json").write_text(json.dumps({**receipt, key: value}))
                    with self.assertRaises(ValueError): study.verify_dry_run(root / "RECEIPT.json", FIXTURE_FREEZE)

    def test_complete_bundle_regenerates_exact_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_bundle(root)
            study.verify_dry_run(root / "RECEIPT.json", FIXTURE_FREEZE)
            with patch.object(study, "check_freeze", return_value=(FIXTURE_FREEZE, {})):
                summary = study.regenerate(root / "RECEIPT.json", root / "regenerated")
                with self.assertRaises(FileExistsError): study.regenerate(root / "RECEIPT.json", root / "regenerated")
            for name in ("SUMMARY.json", "SUMMARY.csv", "RESULTS.md"):
                self.assertEqual((root / name).read_bytes(), (root / "regenerated" / name).read_bytes())
            proof = json.loads((root / "regenerated/REGENERATION.json").read_text())
            self.assertEqual(proof["status"], "PASS_REGENERATED_OUTPUTS")
            self.assertLess(summary["primary_anchor8_minus_direct"], 0)
            self.assertEqual(summary["phase"], "dry-run")
            self.assertEqual(summary["development_screen"], "NOT_EVALUATED_DRY_RUN")

    def test_bad_receipt_refused_before_raw_access(self):
        mutations = {"status": "BLOCKED_RESOURCE", "phase": "confirmation", "version": "OTHER",
                     "data_payload_opened": False, "protected_outcomes_opened": True,
                     "scientific_disposition": "SUPPORTED", "freeze_sha256": "f" * 64,
                     "data_sha256": "f" * 64, "native_source_sha256": "f" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); receipt = fixture_bundle(root)
            for key, value in mutations.items():
                with self.subTest(key=key):
                    changed = {**receipt, key: value}
                    (root / "RECEIPT.json").write_text(json.dumps(changed))
                    with patch.object(evidence, "verified_lines", side_effect=AssertionError("must not open raw")):
                        with self.assertRaises(ValueError): study.verify_dry_run(root / "RECEIPT.json", FIXTURE_FREEZE)

    def test_count_manifest_and_schedule_mismatches_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); receipt = fixture_bundle(root)
            for key, value in (("raw_records", 319), ("query_groups", True), ("noises", [.25]),
                               ("outputs", {"RAW.jsonl": "0" * 64}), ("retained_files", {}),
                               ("query_ids", receipt["query_ids"][:-1])):
                with self.subTest(key=key):
                    (root / "RECEIPT.json").write_text(json.dumps({**receipt, key: value}))
                    with self.assertRaises(ValueError): study.verify_dry_run(root / "RECEIPT.json", FIXTURE_FREEZE)

    def test_coherently_rehashed_wrong_summary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); receipt = fixture_bundle(root)
            value = json.loads((root / "SUMMARY.json").read_text()); value["primary_anchor8_minus_direct"] = 1.
            rewrite(root, receipt, "SUMMARY.json", value)
            with self.assertRaisesRegex(ValueError, "summary disagrees"):
                study.verify_dry_run(root / "RECEIPT.json", FIXTURE_FREEZE)

    def test_coherently_rehashed_wrong_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); receipt = fixture_bundle(root)
            value = json.loads((root / "MODEL.json").read_text()); value["config"]["steps"] = 1
            rewrite(root, receipt, "MODEL.json", value)
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                study.verify_dry_run(root / "RECEIPT.json", FIXTURE_FREEZE)

    def test_split_overlap_and_wrong_role_rejected(self):
        for field in ("overlap", "role"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); receipt = fixture_bundle(root)
                value = json.loads((root / "SPLIT.json").read_text())
                if field == "overlap": value["training_ids"][0] = "digits:0"
                else: value["role"] = "protected_confirmation"
                rewrite(root, receipt, "SPLIT.json", value)
                with self.assertRaises(ValueError): study.verify_dry_run(root / "RECEIPT.json", FIXTURE_FREEZE)

    def test_coherently_rehashed_table_mismatch_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); receipt = fixture_bundle(root)
            rewrite(root, receipt, "SUMMARY.csv", {"wrong": "table"})
            with patch.object(study, "check_freeze", return_value=(FIXTURE_FREEZE, {})):
                with self.assertRaisesRegex(ValueError, "stored tables differ"):
                    study.regenerate(root / "RECEIPT.json", root / "refused")
            self.assertFalse((root / "refused").exists())

    def test_raw_hash_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_bundle(root)
            with (root / "RAW.jsonl").open("ab") as out: out.write(b" ")
            with self.assertRaises(ValueError): study.verify_dry_run(root / "RECEIPT.json", FIXTURE_FREEZE)

    def test_streaming_limits_duplicates_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows"
            for raw, kwargs in ((b'{"x":1,"x":2}\n', {}), (b'{"x":NaN}\n', {}),
                                (b'\n', {}), (b'{"x":1}\n', {"line_limit": 3}),
                                (b'{"x":1}\n', {"limit": 3})):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): list(evidence.verified_lines(path, study.sha(raw), **kwargs))
            target = Path(tmp) / "link"; target.symlink_to(path)
            with self.assertRaises(ValueError): list(evidence.verified_lines(target, study.sha(path.read_bytes())))

    def test_streaming_detects_replacement_during_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "rows"
            raw = b'{"x":1}\n{"x":2}\n'; path.write_bytes(raw)
            stream = evidence.verified_lines(path, study.sha(raw)); next(stream)
            other = root / "replacement"; other.write_bytes(raw); other.replace(path)
            with self.assertRaisesRegex(ValueError, "changed while reading"): list(stream)


if __name__ == "__main__": unittest.main()
