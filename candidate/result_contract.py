"""Streaming QA1 analysis and completed-run evidence validation.

These checks establish internal consistency, not authenticity, independence or
scientific success. No dataset loader or experiment execution lives here.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from safe_io import contained_path, read_bytes, unique_object

METHODS = ("direct", "native_full", "native_one_step", "native_uniform", "anchor8",
           "anchor1", "anchor_alpha0", "direct_repeat8", "knn1", "knn3")
NOISES = (0., .25, .5)
OUTPUT_NAMES = frozenset({"RAW.jsonl", "SUMMARY.json", "SUMMARY.csv", "RESULTS.md",
                          "SPLIT.json", "MODEL.json", "BOUNDS.json",
                          "NATIVE_DIAGNOSTICS.json", "ANCHOR_TRACES.json"})
HEX256 = re.compile(r"[0-9a-f]{64}")


def strict_json(raw):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject)


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {name}")
    return value


def noise_value(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value not in NOISES:
        raise ValueError("invalid declared noise level")
    return value


class SummaryAccumulator:
    """Keep compact pair outcomes and integer totals, never complete raw rows."""

    def __init__(self, expected_queries=None, noises=None, *, phase="development"):
        if phase not in {"dry-run", "development"}:
            raise ValueError("analysis requires a dry-run or development phase")
        self.phase = phase
        self.expected_queries = self._unique(expected_queries, "query")
        self.noises = self._unique(noises, "noise")
        if self.expected_queries is not None:
            if any(not isinstance(q, str) or not q or len(q) > 80 for q in self.expected_queries):
                raise ValueError("invalid expected query ID")
        if self.noises is not None:
            for noise in self.noises:
                noise_value(noise)
        self.expected_query_set = set(self.expected_queries) if self.expected_queries is not None else None
        self.noise_set = set(self.noises) if self.noises is not None else None
        self.labels = {}
        self.cells = {}
        self.groups = {}
        self.count = 0

    @staticmethod
    def _unique(values, name):
        if values is None:
            return None
        if not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values):
            raise ValueError(f"empty or duplicate {name} declarations")
        return tuple(values)

    def add(self, row):
        if not isinstance(row, dict):
            raise ValueError("result row must be an object")
        q, noise, method = row["query_id"], noise_value(row["noise"]), row["method"]
        if not isinstance(q, str) or not q or len(q) > 80 or method not in METHODS:
            raise ValueError("invalid query or method")
        if self.expected_queries is not None and q not in self.expected_query_set:
            raise ValueError("unexpected query")
        if self.noises is not None and noise not in self.noise_set:
            raise ValueError("unexpected noise")
        label = integer(row["label"], 0, 9, "label")
        prediction = integer(row["prediction"], 0, 9, "prediction")
        initial = integer(row["initial_prediction"], 0, 9, "initial prediction")
        elapsed = integer(row["elapsed_ns"], 0, 600_000_000_000, "elapsed time")
        if q in self.labels and self.labels[q] != label:
            raise ValueError("query label changes across methods or noise conditions")
        key = q, noise, method
        if key in self.cells:
            raise ValueError("duplicate result cell")
        self.labels[q] = label
        self.cells[key] = prediction, initial
        group = self.groups.setdefault((noise, method), [0, 0, 0, Counter()])
        group[0] += 1
        group[1] += prediction == label
        group[2] += elapsed
        group[3][prediction] += 1
        self.count += 1

    def finish(self):
        if not self.count:
            raise ValueError("empty results")
        ids = set(self.expected_queries if self.expected_queries is not None else self.labels)
        levels = set(self.noises if self.noises is not None else (key[1] for key in self.cells))
        if self.phase == "dry-run" and levels != {0.}:
            raise ValueError("dry-run permits clean queries only")
        if self.count != len(ids) * len(levels) * len(METHODS):
            raise ValueError("incomplete or extra result matrix")
        table, contrasts = [], []
        for (noise, method), (n, correct, elapsed, counts) in sorted(self.groups.items()):
            table.append({"noise": noise, "method": method, "n": n, "correct": correct,
                          "accuracy": correct / n, "winner_counts": dict(sorted(counts.items())),
                          "largest_winner_fraction": max(counts.values()) / n,
                          "mean_elapsed_ns": float(elapsed) / n})
        for noise in sorted(levels):
            for method in METHODS:
                better = worse = changed = 0
                for q in sorted(ids):
                    if (q, noise, method) not in self.cells or (q, noise, "direct") not in self.cells:
                        raise ValueError("incomplete result matrix")
                    direct, direct_initial = self.cells[q, noise, "direct"]
                    prediction, initial = self.cells[q, noise, method]
                    if initial != direct or direct_initial != direct:
                        raise ValueError("recorded initial prediction differs from direct")
                    if method == "direct_repeat8" and prediction != direct:
                        raise ValueError("repeated direct control changed prediction")
                    bc, cc = direct == self.labels[q], prediction == self.labels[q]
                    better += cc and not bc
                    worse += bc and not cc
                    changed += prediction != direct
                contrasts.append({"noise": noise, "method": method, "n": len(ids),
                                  "beneficial_changes": better, "harmful_changes": worse,
                                  "prediction_changes": changed,
                                  "accuracy_delta_vs_direct": (better - worse) / len(ids)})
        primary = [row["accuracy_delta_vs_direct"] for row in contrasts if row["method"] == "anchor8"]
        numerator = sum(row["beneficial_changes"] - row["harmful_changes"]
                        for row in contrasts if row["method"] == "anchor8")
        denominator = len(ids) * len(levels)
        average = numerator / denominator
        screen = "NOMINATE_FOR_FRESH_CONFIRMATION" if 100 * numerator >= denominator else "UTILITY_SCREEN_NOT_MET"
        if self.phase == "dry-run":
            screen = "NOT_EVALUATED_DRY_RUN"
        return {"status": "COMPLETE_EXPOSED_DEVELOPMENT_ANALYSIS", "table": table, "paired": contrasts,
                "phase": self.phase, "primary_exact_ratio": {"numerator": numerator, "denominator": denominator},
                "primary_anchor8_minus_direct": average, "noise_condition_envelope": [min(primary), max(primary)],
                "development_screen": screen,
                "uncertainty": "Finite-cohort discordances and noise sensitivity only; no independent-writer or population confidence claim.",
                "scientific_disposition": "NOT_CONFIRMATORY", "accuracy_is_new_only_if_run_completed": True}


def summarize(records, expected_queries=None, noises=None, *, phase="development"):
    accumulator = SummaryAccumulator(expected_queries, noises, phase=phase)
    for row in records:
        accumulator.add(row)
    return accumulator.finish()


@contextmanager
def stable_file(path, limit):
    path = Path(path)
    contained_path(path.parent, path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("invalid or oversized artifact")
        yield handle
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if identity(before) != identity(os.fstat(handle.fileno())) or identity(before) != identity(path.stat(follow_symlinks=False)):
            raise ValueError("artifact changed while reading")


def verified_lines(path, expected_sha, limit=20_000_000, line_limit=65_536):
    digest = hashlib.sha256()
    total = 0
    with stable_file(path, limit) as handle:
        while line := handle.readline(line_limit + 1):
            total += len(line)
            if len(line) > line_limit or total > limit or not line.strip():
                raise ValueError("oversized or blank JSONL record")
            digest.update(line)
            yield strict_json(line)
        if digest.hexdigest() != expected_sha:
            raise ValueError("raw artifact changed")


def file_hash(path, limit=20_000_000):
    digest = hashlib.sha256()
    total = 0
    with stable_file(path, limit) as handle:
        while block := handle.read(65_536):
            total += len(block)
            if total > limit:
                raise ValueError("artifact grew beyond limit")
            digest.update(block)
    return digest.hexdigest()


def digit_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"digits:(0|[1-9][0-9]*)", value):
        raise ValueError("invalid digits row ID")
    integer(int(value.split(":")[1]), 0, 1796, "digits row index")
    return value


def read_verified_run(
    receipt_path,
    freeze_hash,
    data_hash,
    reference_hash,
    expected_phase=None,
    *,
    expected_engineering_revision=None,
    expected_recovery_version=None,
):
    """Return a consistent completed QA1 receipt, summary and model checkpoint.

    Callers must first verify the local package freeze. No claim of externally
    authenticated execution follows from locally editable hashes and receipts.
    """
    receipt_path = Path(receipt_path)
    root = receipt_path.parent
    receipt = strict_json(read_bytes(receipt_path, 1_000_000))
    if (receipt.get("version") != "EPU-QA1" or receipt.get("status") != "COMPLETE_EXPOSED_DEVELOPMENT"
            or receipt.get("phase") not in {"dry-run", "development"}
            or receipt.get("scientific_disposition") != "NOT_CONFIRMATORY"
            or receipt.get("data_payload_opened") is not True
            or receipt.get("protected_outcomes_opened") is not False
            or receipt.get("freeze_sha256") != freeze_hash
            or receipt.get("data_sha256") != data_hash
            or receipt.get("native_source_sha256") != reference_hash):
        raise ValueError("unadmitted or mismatched completed-run receipt")
    if expected_phase is not None and receipt["phase"] != expected_phase:
        raise ValueError("wrong prerequisite phase")
    if (expected_engineering_revision is not None
            and receipt.get("engineering_revision") != expected_engineering_revision):
        raise ValueError("wrong engineering revision")
    if (expected_recovery_version is not None
            and receipt.get("recovery_version") != expected_recovery_version):
        raise ValueError("wrong recovery version")
    if receipt["phase"] == "development" and expected_recovery_version is not None:
        ledgers = receipt.get("attempt_budget_ledgers")
        admission = receipt.get("recovery_admission")
        if (not isinstance(ledgers, list) or len(ledgers) != 1
                or not isinstance(ledgers[0], str) or not ledgers[0]):
            raise ValueError("missing unique recovery attempt ledger binding")
        if (not isinstance(admission, dict) or admission.get("status") != "PASS"
                or admission.get("floor_bytes") != 1024**3
                or admission.get("minimum_span_seconds") != 30
                or admission.get("minimum_samples") != 31):
            raise ValueError("missing sustained recovery admission")
        samples = admission.get("samples")
        if not isinstance(samples, list) or len(samples) != 31:
            raise ValueError("incomplete sustained recovery admission")
        if (type(samples[0].get("elapsed_seconds")) not in (int, float)
                or type(samples[-1].get("elapsed_seconds")) not in (int, float)
                or samples[-1]["elapsed_seconds"] - samples[0]["elapsed_seconds"] < 30):
            raise ValueError("short sustained recovery admission")
    elapsed = receipt.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= 600:
        raise ValueError("invalid or exceeded recorded wall budget")
    integer(receipt.get("max_rss_bytes"), 1, 2**63 - 1, "recorded peak memory")
    checks = receipt.get("resource_checks")
    if not isinstance(checks, list) or len(checks) < 2:
        raise ValueError("missing admission and completion resource checks")
    for check in checks:
        if check.get("status") != "PASS" or check.get("floor_bytes") != 536870912:
            raise ValueError("failed resource check in completed run")
        samples = check.get("samples")
        if not isinstance(samples, list) or len(samples) != 2:
            raise ValueError("incomplete resource samples")
        for sample in samples:
            integer(sample.get("free_plus_speculative_bytes"), 536870912, 2**63 - 1, "free memory")
            integer(sample.get("pageouts"), 0, 2**63 - 1, "pageouts")
            integer(sample.get("swap_used_bytes"), 0, 2**63 - 1, "swap usage")
        if samples[1]["pageouts"] != samples[0]["pageouts"] or samples[1]["swap_used_bytes"] > samples[0]["swap_used_bytes"]:
            raise ValueError("unstable resource samples in completed run")
    queries, noises = receipt["query_ids"], receipt["noises"]
    if not isinstance(queries, list) or not queries or len(set(queries)) != len(queries):
        raise ValueError("invalid query declaration")
    for query in queries:
        digit_id(query)
    expected_noises = [0.] if receipt["phase"] == "dry-run" else list(NOISES)
    if not isinstance(noises, list) or any(type(x) not in (int, float) for x in noises) or noises != expected_noises:
        raise ValueError("phase/noise schedule mismatch")
    if (receipt["phase"] == "dry-run" and len(queries) != 32) or len(queries) > 450:
        raise ValueError("phase/query count mismatch")
    integer(receipt["query_groups"], len(queries), len(queries), "query group count")
    integer(receipt["raw_records"], len(queries) * len(noises) * len(METHODS),
            len(queries) * len(noises) * len(METHODS), "raw record count")
    outputs = receipt["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != OUTPUT_NAMES:
        raise ValueError("missing or unexpected output manifest entries")
    retained = receipt.get("retained_files")
    if not isinstance(retained, dict) or set(retained) != OUTPUT_NAMES | {"STARTED.json"}:
        raise ValueError("incomplete retained-file manifest")
    for name, digest in retained.items():
        if not isinstance(digest, str) or not HEX256.fullmatch(digest):
            raise ValueError("invalid artifact digest")
        if name in outputs and outputs[name] != digest:
            raise ValueError("output and retained manifests disagree")
        if name != "RAW.jsonl" and file_hash(root / name) != digest:
            raise ValueError(f"artifact changed: {name}")
    split = strict_json(read_bytes(root / "SPLIT.json", 1_000_000))
    if split.get("role") != "historically_exposed_development" or split.get("query_ids") != queries:
        raise ValueError("split role or selected queries disagree")
    training, aliases = split["training_ids"], split["all_query_aliases"]
    if not isinstance(training, list) or not isinstance(aliases, dict) or not set(queries) <= set(aliases):
        raise ValueError("invalid split membership")
    members = list(training)
    for representative, group in aliases.items():
        digit_id(representative)
        if not isinstance(group, list) or not group or group[0] != representative:
            raise ValueError("invalid alias group")
        members.extend(group)
    if len(members) != 1797 or len(set(members)) != 1797 or set(members) != {f"digits:{i}" for i in range(1797)}:
        raise ValueError("split does not partition dataset row IDs")
    if sum(len(group) for group in aliases.values()) < 450:
        raise ValueError("undersized development query allocation")
    if receipt["phase"] == "development" and set(queries) != set(aliases):
        raise ValueError("incomplete development query groups")
    integer(receipt["training_rows"], len(training), len(training), "training row count")
    summary = summarize(verified_lines(root / "RAW.jsonl", outputs["RAW.jsonl"]), queries, noises, phase=receipt["phase"])
    stored = strict_json(read_bytes(root / "SUMMARY.json", 1_000_000))
    if json.loads(json.dumps(summary, allow_nan=False)) != stored:
        raise ValueError("stored summary disagrees with raw records")
    model = strict_json(read_bytes(root / "MODEL.json", 1_000_000))
    return receipt, summary, model
