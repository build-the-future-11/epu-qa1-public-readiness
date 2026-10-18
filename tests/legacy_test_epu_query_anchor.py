"""Small engineering fixtures, never evidence of dataset efficacy."""
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import epu_diagnostics as diagnostics
from epu_query_anchor import Config, PrototypeModel, NeighborBank, checked_state, dot, fit_class_means, softmax, unit_center
import run_study as study

NATIVE = Path(os.environ.get("EPU_REFERENCE_ROOT", str(Path(__file__).resolve().parents[2])))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.bank = [unit_center([1, 2, 0, 0]), unit_center([0, 1, 3, 1]), unit_center([3, 0, 1, 0])]
        self.q = unit_center([2, 1, 0, 2])
        self.model = PrototypeModel(self.bank, [0, 1, 2])

    def test_native_lambda_one_state_and_scores_parity(self):
        reference = diagnostics.load_reference(NATIVE)
        model = PrototypeModel(self.bank, [0, 1, 2], Config(memory_mix=1.))
        core = reference(dim=4, memories=self.bank, z=list(self.q), alpha=.6, beta=6., threshold=0.)
        core.run(steps=8)
        result = model.predict(self.q)
        self.assertLess(math.dist(core.z, result["state"]), 1e-12)
        self.assertEqual(result["prediction"], self.model.decode(unit_center(core.z))["prediction"])

    def test_lambda_zero_and_zero_steps_preserve_query(self):
        for config in (Config(memory_mix=0.), Config(steps=0)):
            result = PrototypeModel(self.bank, [0, 1, 2], config).predict(self.q)
            self.assertLess(math.dist(result["state"], self.q), 1e-14)
            self.assertEqual(result["prediction"], self.model.decode(self.q)["prediction"])

    def test_exact_one_memory_recurrence(self):
        m = self.bank[0]
        result = PrototypeModel([m], [4]).predict(self.q)
        scale = .1 * (1 - .6 ** 8)
        expected = [(1 - scale) * q + scale * v for q, v in zip(self.q, m)]
        self.assertLess(math.dist(result["state"], expected), 1e-14)
        self.assertEqual(result["prediction"], 4)

    def test_energy_descent_drift_and_invariant_domain(self):
        for step in self.model.predict(self.q, trace=True)["trace"]:
            checked_state(step["state"], 4)
            self.assertLessEqual(step["energy"] - step["energy_before"], -2 * step["delta_l2"] ** 2 + 1e-13)
            self.assertLessEqual(step["query_distance"], .2 * (1 - .6 ** step["step"]) + 1e-13)
        self.assertLessEqual(self.model.bounds()["update_lipschitz_bound"], .84 + 1e-13)

    def test_contraction_with_query_fixed(self):
        # Independent one-step formula; query must stay fixed for this bound.
        def update(z):
            weights = softmax([6 * dot(z, m) for m in self.bank])
            r = [sum(w * m[j] for w, m in zip(weights, self.bank)) for j in range(4)]
            return [.6 * z[j] + .4 * (.9 * self.q[j] + .1 * r[j]) for j in range(4)]
        x, y = self.bank[:2]
        self.assertLessEqual(math.dist(update(x), update(y)), self.model.bounds()["update_lipschitz_bound"] * math.dist(x, y) + 1e-13)

    def test_finite_difference_energy_gradient(self):
        lam = .1
        state = [x * .5 for x in self.q]
        w = softmax([6 * dot(state, m) for m in self.bank])
        recon = [sum(wi * m[j] for wi, m in zip(w, self.bank)) for j in range(4)]
        gradient = [state[j] - .9 * self.q[j] - lam * recon[j] for j in range(4)]
        for j in range(3):
            plus, minus = list(state), list(state)
            plus[j] += 1e-6; minus[j] -= 1e-6
            plus[-1] -= 1e-6; minus[-1] += 1e-6
            derivative = (self.model.energy(plus, self.q) - self.model.energy(minus, self.q)) / 2e-6
            self.assertAlmostEqual(derivative, gradient[j] - gradient[-1], places=8)

    def test_permuting_memories_preserves_untied_label(self):
        perm = [2, 0, 1]
        changed = PrototypeModel([self.bank[i] for i in perm], perm)
        a, b = self.model.predict(self.q), changed.predict(self.q)
        self.assertEqual(a["prediction"], b["prediction"])
        self.assertLess(math.dist(a["state"], b["state"]), 1e-13)

    def test_zero_and_duplicate_ties_are_visible(self):
        model = PrototypeModel([[0., 0.], [0., 0.]], [7, 2])
        result = model.predict([0., 0.])
        self.assertEqual(result["prediction"], 7)
        self.assertEqual(result["exact_tie_indices"], [0, 1])
        self.assertEqual(result["state_norm"], 0)

    def test_stable_softmax_handles_extreme_shift(self):
        self.assertEqual(softmax([10000., 10000.]), (.5, .5))
        self.assertAlmostEqual(sum(softmax([10000., -10000.])), 1.)

    def test_invalid_domains_fail(self):
        for values in ([True, 0], [float("nan"), 0], [float("inf"), 0], [1], [2, -2], [.2, .2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                checked_state(values)
        for kwargs in ({"alpha":1.}, {"alpha":True}, {"beta":0}, {"memory_mix":-1}, {"steps":True}, {"steps":65}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        with self.assertRaises(ValueError):
            dot([1, 2], [1])

    def test_train_means_and_class_order(self):
        trained = fit_class_means([[1, 0, 0], [0, 1, 0], [.5, 0, .5]], [2, 1, 2])
        self.assertEqual(trained.labels, (1, 2))
        self.assertEqual(trained.memories[1], unit_center([.75, 0, .25]))
        with self.assertRaises(ValueError):
            fit_class_means([[0, 1]], [True])

    def test_checkpoint_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            loaded = PrototypeModel.from_dict(json.loads(path.read_text()))
            self.assertEqual(loaded.predict(self.q), self.model.predict(self.q))
            with self.assertRaises(FileExistsError):
                loaded.save(path)
        value = self.model.to_dict(); value["gamma"] = 0
        with self.assertRaises(ValueError):
            PrototypeModel.from_dict(value)

    def test_public_scores_and_energy_reject_invalid_state(self):
        for method in (self.model.scores, lambda x: self.model.energy(x, self.q)):
            with self.assertRaises(ValueError):
                method([float("nan"), 0, 0, 0])
            with self.assertRaises(ValueError):
                method([2, -2, 0, 0])

    def test_prepared_neighbors_match_full_sort_with_ties(self):
        rng = random.Random(71)
        bank = [unit_center([rng.random() for _ in range(4)]) for _ in range(21)]
        bank += [bank[0], bank[1]]
        labels = [i % 4 for i in range(len(bank))]
        indices = list(reversed(range(len(bank))))
        prepared = NeighborBank(bank, labels, indices)
        for q in [self.q, bank[0], (0., 0., 0., 0.)]:
            sorted_all = sorted((-dot(q, m), i, y) for m, i, y in zip(bank, indices, labels))
            votes = {}
            for _, _, y in sorted_all[:3]:
                votes[y] = votes.get(y, 0) + 1
            result = prepared.predict(q)
            self.assertEqual(result["knn1"], sorted_all[0][2])
            self.assertEqual(result["knn3"], min(votes, key=lambda y: (-votes[y], y)))
        bank[0] = (0., 0., 0., 0.)
        self.assertNotEqual(prepared.training[0], bank[0])
        with self.assertRaises(ValueError):
            prepared.predict([True, 0, 0, 0])


class ExperimentTests(unittest.TestCase):
    def fixture_rows(self):
        return [{"id": f"digits:{i}", "row_index": i, "pixels": (i % 17, (i + 1) % 17), "label": i % 2} for i in range(12)]

    def fixture_records(self):
        rows = []
        for q in ("a", "b"):
            for method in study.METHODS:
                rows.append({"query_id": q, "noise": 0., "method": method, "label": 0,
                             "prediction": int(q == "a" and method != "anchor8"), "initial_prediction": int(q == "a"), "elapsed_ns": 1})
        return rows

    def test_duplicate_groups_never_cross_split(self):
        rows = self.fixture_rows()
        rows.append({**rows[0], "id": "digits:12", "row_index": 12})
        train, query, aliases = study.grouped_split(rows, target_query_rows=4)
        self.assertFalse({r["pixels"] for r in train} & {r["pixels"] for r in query})
        all_ids = [r["id"] for r in train] + [i for ids in aliases.values() for i in ids]
        self.assertEqual(len(set(all_ids)), len(rows))
        self.assertEqual(study.grouped_split(list(reversed(rows)), 4), (train, query, aliases))

    def test_conflicting_duplicate_label_refuses(self):
        rows = self.fixture_rows()
        rows.append({**rows[0], "id": "digits:12", "row_index": 12, "label": 1})
        with self.assertRaises(ValueError):
            study.grouped_split(rows, 4)

    def test_noise_is_reproducible_and_query_specific(self):
        rows = self.fixture_rows()
        self.assertEqual(study.corrupted_query(rows[0], .25), study.corrupted_query(rows[0], .25))
        self.assertEqual(study.corrupted_query(rows[0], 0.), unit_center([v / 16 for v in rows[0]["pixels"]]))

    def test_paired_summary_and_incomplete_refusal(self):
        rows = self.fixture_records()
        result = study.summarize(rows, ["a", "b"], [0.])
        self.assertEqual(result["primary_anchor8_minus_direct"], .5)
        contrast = next(r for r in result["paired"] if r["method"] == "anchor8")
        self.assertEqual(contrast["beneficial_changes"], 1)
        self.assertEqual(contrast["harmful_changes"], 0)
        with self.assertRaises(ValueError):
            study.summarize(rows[:-1], ["a", "b"], [0.])
        with self.assertRaises(ValueError):
            study.summarize(rows + rows[:1])

    def test_native_and_candidate_matrix_executes(self):
        reference = diagnostics.load_reference(NATIVE)
        bank = [unit_center([1, 0, 0]), unit_center([0, 1, 0]), unit_center([0, 0, 1])]
        models = {"anchor8": PrototypeModel(bank, [0,1,2]),
                  "anchor1": PrototypeModel(bank, [0,1,2], Config(steps=1)),
                  "anchor_alpha0": PrototypeModel(bank, [0,1,2], Config(alpha=0.))}
        results = study.evaluate_query(reference, models, bank[0], bank, [0,1,2], [0,1,2])
        self.assertEqual(set(results), set(study.METHODS))
        self.assertEqual(results["direct"]["prediction"], results["direct_repeat8"]["prediction"])
        self.assertEqual(results["knn1"]["prediction"], 0)
        self.assertEqual(results["knn3"]["prediction"], 0)

    def test_resource_refusal_precedes_data_and_budget(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, "check_freeze", return_value=("fixed", {})), \
             patch.object(diagnostics, "load_reference"), \
             patch.object(diagnostics, "resource_admission", return_value={"status":"BLOCKED_RESOURCE"}), \
             patch.object(study, "load_digits_file", side_effect=AssertionError("must not open data")):
            root = Path(directory)
            code, receipt, output = study.run("dry-run", "absent", NATIVE, "blocked", package=root)
            self.assertEqual(code, 2)
            self.assertFalse(receipt["data_payload_opened"])
            self.assertEqual(receipt["status"], "BLOCKED_RESOURCE")
            self.assertTrue((output / "RECEIPT.json").is_file())
            self.assertFalse((root / "runs/ATTEMPTS.jsonl").exists())
            with self.assertRaises(FileExistsError):
                study.run("dry-run", "absent", NATIVE, "blocked", package=root)

    def test_measurement_failure_refuses_and_keeps_error(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, "check_freeze", return_value=("fixed", {})), \
             patch.object(diagnostics, "load_reference"), \
             patch.object(diagnostics, "resource_admission", side_effect=OSError("measurement unavailable")):
            code, receipt, output = study.run("dry-run", "absent", NATIVE, "failed", package=Path(directory))
            self.assertEqual(code, 2)
            self.assertFalse(receipt["data_payload_opened"])
            self.assertTrue((output / "ERROR.log").exists())

    def test_data_failure_consumes_attempt_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, "check_freeze", return_value=("fixed", {})), \
             patch.object(diagnostics, "load_reference"), \
             patch.object(diagnostics, "resource_admission", return_value={"status":"PASS"}), \
             patch.object(study, "load_digits_file", side_effect=ValueError("wrong dataset hash")) as loader:
            root = Path(directory)
            first = study.run("dry-run", "absent", NATIVE, "first", package=root)
            second = study.run("dry-run", "absent", NATIVE, "second", package=root)
            self.assertEqual(loader.call_count, 1)
            self.assertTrue(first[1]["data_payload_opened"])
            self.assertFalse(second[1]["data_payload_opened"])
            self.assertIn("budget exhausted", second[1]["error"])

    def test_full_run_requires_matching_real_dry_run(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study, "check_freeze", return_value=("fixed", {})), \
             patch.object(diagnostics, "load_reference"), \
             patch.object(diagnostics, "resource_admission", return_value={"status":"PASS"}):
            code, receipt, _ = study.run("development", "absent", NATIVE, "without-dry", package=Path(directory))
            self.assertEqual(code, 2)
            self.assertFalse(receipt["data_payload_opened"])

    def test_raw_regeneration_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b"\n".join(json.dumps(r).encode() for r in self.fixture_records()) + b"\n"
            (root / "RAW.jsonl").write_bytes(raw)
            receipt = {"status":"COMPLETE_EXPOSED_DEVELOPMENT", "query_ids":["a","b"], "noises":[0.],
                       "outputs":{"RAW.jsonl":study.sha(raw)}}
            study.dump_new(root / "RECEIPT.json", receipt)
            with patch.object(study, "check_freeze", return_value=("fixed", {})), self.assertRaises(ValueError):
                study.regenerate(root / "RECEIPT.json", root / "regenerated")
            self.assertFalse((root / "regenerated").exists())
            (root / "RAW.jsonl").write_bytes(raw + b" ")
            with patch.object(study, "check_freeze", return_value=("fixed", {})), self.assertRaises(ValueError):
                study.regenerate(root / "RECEIPT.json", root / "bad")

    def test_confirmation_mode_and_bad_run_ids_refuse(self):
        for phase, run_id in (("confirmation", "x"), ("dry-run", "../escape")):
            with self.assertRaises(ValueError):
                study.run(phase, "absent", NATIVE, run_id)

    def test_scientific_source_change_breaks_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("run_study.py", "epu_query_anchor.py", "epu_diagnostics.py", "safe_io.py", "result_contract.py", "analyze_results.py", "SCIENTIFIC_SPEC.md", "PROTOCOL.md")
            for name in names:
                (root / name).write_text("fixture")
            freeze = {"version":"EPU-QA1", "role":"exposed_development_only", "engineering_tests":"PASS",
                      "files":{name:study.sha(b"fixture") for name in names}}
            study.dump_new(root / "FREEZE.json", freeze)
            self.assertTrue(study.check_freeze(root)[0])
            (root / "PROTOCOL.md").write_text("altered")
            with self.assertRaises(ValueError):
                study.check_freeze(root)

    def test_dry_run_tampering_and_metadata_only_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("RAW.jsonl", "SUMMARY.json", "SPLIT.json", "MODEL.json")
            for name in names:
                (root / name).write_text("fixture")
            receipt = {"status":"COMPLETE_EXPOSED_DEVELOPMENT", "phase":"dry-run", "freeze_sha256":"fixed",
                       "data_sha256":study.DATA_SHA256, "outputs":{n:study.sha(b"fixture") for n in names}}
            study.dump_new(root / "RECEIPT.json", receipt)
            with self.assertRaises(ValueError):
                study.verify_dry_run(root / "RECEIPT.json", "fixed")
            (root / "RAW.jsonl").write_text("changed")
            with self.assertRaises(ValueError):
                study.verify_dry_run(root / "RECEIPT.json", "fixed")
            receipt["status"] = "PASS_METADATA_ONLY"
            study.dump_new(root / "metadata.json", receipt)
            with self.assertRaises(ValueError):
                study.verify_dry_run(root / "metadata.json", "fixed")


if __name__ == "__main__":
    unittest.main()
