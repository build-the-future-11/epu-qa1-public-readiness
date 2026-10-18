#!/usr/bin/env python3
"""Guarded, finite-budget EPU-QA1 exposed-data development study. See PROTOCOL.md."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path
import random
import re
import resource
import time
import traceback

import epu_diagnostics as diagnostics
import result_contract as evidence
import recovery_budget as recovery
from result_contract import METHODS, SummaryAccumulator
from epu_query_anchor import Config, PrototypeModel, NeighborBank, fit_class_means, nearest_neighbors, unit_center

DATA_SHA256 = "09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22"
DATA_BYTES = 57523
PACKAGE = Path(__file__).resolve().parent

# Retain the established public analysis entry point while keeping the
# implementation in the evidence-only contract module.
summarize = evidence.summarize


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, allow_nan=False, sort_keys=True)
        out.write("\n")


def read_small(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"invalid or oversized input: {path}")
    with path.open("rb") as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise ValueError("input grew beyond limit")
    return value


def check_freeze(root=PACKAGE):
    raw = read_small(root / "FREEZE.json", 100_000)
    freeze = diagnostics.strict_json(raw)
    if (freeze.get("version") != "EPU-QA1" or freeze.get("role") != "exposed_development_only"
            or freeze.get("engineering_tests") != "PASS" or not freeze.get("files")):
        raise ValueError("unadmitted package freeze")
    required = {"recovery_budget.py", "RECOVERY_AMENDMENT.md", "run_study.py", "epu_query_anchor.py", "epu_diagnostics.py", "safe_io.py", "result_contract.py", "analyze_results.py", "SCIENTIFIC_SPEC.md", "PROTOCOL.md"}
    if not required <= set(freeze["files"]):
        raise ValueError("freeze omits scientific source")
    if freeze.get("engineering_revision") in {"E1", "E2", "E3", "E4", "E5"} and (
            freeze.get("shared_attempt_directory") != "successors/query_anchor_v1_1/runs"
            or freeze.get("predecessor_attempt_directories") != ["successors/query_anchor_v1/runs"]):
        raise ValueError("Efficiency revisions must share both predecessor QA1 data-opening ledgers")
    recovery.validate_freeze(freeze)
    for relative, expected in freeze["files"].items():
        if not isinstance(relative, str) or "/" in relative or "\\" in relative or relative.startswith("."):
            raise ValueError("freeze must bind flat package files")
        if sha(read_small(root / relative, 1_000_000)) != expected:
            raise ValueError(f"package changed after freeze: {relative}")
    return sha(raw), freeze


def load_digits_file(path):
    raw = read_small(path, DATA_BYTES)
    if len(raw) != DATA_BYTES or sha(raw) != DATA_SHA256:
        raise ValueError("digits bytes differ from the declared wheel RECORD")
    # Exact trusted digest and size bound precede decompression.
    text = gzip.decompress(raw)
    if len(text) > 4_000_000:
        raise ValueError("unexpected decompressed size")
    rows = []
    for index, fields in enumerate(csv.reader(io.StringIO(text.decode("ascii")))):
        if len(fields) != 65:
            raise ValueError("digits must have 64 pixels plus a label")
        numbers = [float(value) for value in fields]
        if any(not math.isfinite(x) or not x.is_integer() for x in numbers):
            raise ValueError("nonintegral/nonfinite digits value")
        pixels, label = tuple(int(x) for x in numbers[:64]), int(numbers[64])
        if any(x < 0 or x > 16 for x in pixels) or label not in range(10):
            raise ValueError("digits value outside source domain")
        rows.append({"id": f"digits:{index}", "row_index": index, "pixels": pixels, "label": label})
    if len(rows) != 1797:
        raise ValueError("digits row count changed")
    return rows


def grouped_split(rows, target_query_rows=450):
    """The only split policy; groups are ordered without looking at class labels."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row["pixels"]), []).append(row)
    ordered = sorted(groups.items(), key=lambda item: (
        sha(("EPU-QA1/split/" + ",".join(str(x) for x in item[0])).encode("ascii")),
        min(r["row_index"] for r in item[1])))
    train, queries, aliases = [], [], {}
    assigned = 0
    for _pixels, members in ordered:
        if len({row["label"] for row in members}) != 1:
            raise ValueError("identical image has conflicting labels")
        members = sorted(members, key=lambda r: r["row_index"])
        if assigned < target_query_rows:
            representative = members[0]
            queries.append(representative)
            aliases[representative["id"]] = [row["id"] for row in members]
            assigned += len(members)
        else:
            train.extend(members)
    train.sort(key=lambda r: r["row_index"])
    if not train or not queries:
        raise ValueError("empty split")
    if {r["pixels"] for r in train} & {r["pixels"] for r in queries}:
        raise ValueError("feature-group leakage")
    training_ids = [r["id"] for r in train]
    all_query_ids = [i for values in aliases.values() for i in values]
    if len(set(training_ids + all_query_ids)) != len(rows):
        raise ValueError("split does not partition unique row IDs")
    return train, queries, aliases


def corrupted_query(row, noise):
    seed = int(sha(f"EPU-QA1/noise/{row['id']}/{noise}".encode("ascii")), 16)
    rng = random.Random(seed)
    values = [max(0., min(1., x / 16 + rng.gauss(0., noise))) for x in row["pixels"]]
    return unit_center(values)


def native_prediction(reference, model, query, *, steps, mode="softmax"):
    core = reference(dim=len(query), beta=6., alpha=.6, threshold=0.,
                             memories=[list(m) for m in model.memories], z=list(query), weight_mode=mode)
    history = core.run(steps=steps)
    # Reproduce native benchmark's final normalization explicitly, including ties.
    final = unit_center(core.z)
    decoded = model.decode(final)
    decoded.update({"state_norm": math.hypot(*core.z), "query_distance": math.dist(query, core.z),
                    "retrieval_score_dot_products": len(history) * len(model.memories),
                    "decoder_dot_products": len(model.memories)})
    return decoded


def evaluate_query(reference, models, query, knn_bank, knn_labels=None, knn_ids=None):
    base = models["anchor8"]
    methods = {
        "direct": lambda: base.decode(query),
        "native_full": lambda: native_prediction(reference, base, query, steps=8),
        "native_one_step": lambda: native_prediction(reference, base, query, steps=1),
        "native_uniform": lambda: native_prediction(reference, base, query, steps=8, mode="uniform"),
        "anchor8": lambda: models["anchor8"].predict(query),
        "anchor1": lambda: models["anchor1"].predict(query),
        "anchor_alpha0": lambda: models["anchor_alpha0"].predict(query),
    }
    results = {}
    for name, method in methods.items():
        start = time.perf_counter_ns()
        result = method()
        result["elapsed_ns"] = time.perf_counter_ns() - start
        results[name] = result
    start = time.perf_counter_ns()
    for _ in range(8):
        repeated = base.decode(query)
    results["direct_repeat8"] = {**repeated, "elapsed_ns": time.perf_counter_ns() - start,
                                 "retrieval_score_dot_products": 0,
                                 "decoder_dot_products": 8 * len(base.memories)}
    results["direct"].update({"retrieval_score_dot_products": 0, "decoder_dot_products": len(base.memories)})
    start = time.perf_counter_ns()
    neighbors = (knn_bank.predict(query) if isinstance(knn_bank, NeighborBank)
                 else nearest_neighbors(query, knn_bank, knn_labels, knn_ids))
    elapsed = time.perf_counter_ns() - start
    for method in ("knn1", "knn3"):
        results[method] = {"prediction": neighbors[method], "elapsed_ns": elapsed,
                           "timing_scope": "shared full-bank scoring and k1/k3 voting; do not sum both times",
                           "retrieval_score_dot_products": len(knn_bank), "decoder_dot_products": 0}
    return results


def table_bytes(summary):
    fields = ["noise", "method", "n", "correct", "accuracy", "largest_winner_fraction", "mean_elapsed_ns"]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary["table"])
    title = "EPU-QA1 dry-run engineering results" if summary["phase"] == "dry-run" else "EPU-QA1 exposed development results"
    lines = [f"# {title}", "", "No confirmatory or population claim.", "",
             "| Noise | Method | Correct / n | Accuracy |", "|---|---|---|---|"]
    lines.extend(
        f"| {row['noise']:g} | {row['method']} | {row['correct']} / {row['n']} | {row['accuracy']:.6f} |"
        for row in summary["table"]
    )
    lines += ["", f"Anchor8 minus direct: {summary['primary_anchor8_minus_direct']:.6f}.",
              f"Development screen: {summary['development_screen']}.", "", summary["uncertainty"]]
    return {"SUMMARY.csv": handle.getvalue().encode("utf-8"),
            "RESULTS.md": ("\n".join(lines) + "\n").encode("utf-8")}


def write_tables(output, summary):
    for name, raw in table_bytes(summary).items():
        with (output / name).open("xb") as handle:
            handle.write(raw)


class ResourceBlocked(RuntimeError):
    pass


def require_time(started):
    """Cooperative deadline: check around stages, never report late success."""
    if time.monotonic() - started > 600:
        raise TimeoutError("600-second attempt budget exceeded")


def require_resources(receipt):
    resources = diagnostics.resource_admission()
    receipt.setdefault("resource_checks", []).append({"at": now(), **resources})
    if resources["status"] != "PASS":
        raise ResourceBlocked("native memory/pageout/swap guard failed")


def verify_completed_run(path, freeze_hash, expected_phase=None):
    recovery_expectations = (
        {
            "expected_engineering_revision": "E5-R1",
            "expected_recovery_version": recovery.RECOVERY_VERSION,
        }
        if expected_phase == "development"
        else {}
    )
    receipt, summary, checkpoint = evidence.read_verified_run(
        path, freeze_hash, DATA_SHA256, diagnostics.REFERENCE_SHA256, expected_phase,
        **recovery_expectations,
    )
    model = PrototypeModel.from_dict(checkpoint)
    if (len(model.memories) != 10 or len(model.memories[0]) != 64
            or model.labels != tuple(range(10)) or model.config != Config()):
        raise ValueError("checkpoint differs from the fixed QA1 model contract")
    rendered = table_bytes(summary)
    rendered["SUMMARY.json"] = (json.dumps(summary, indent=2, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    if any(sha(raw) != receipt["outputs"][name] for name, raw in rendered.items()):
        raise ValueError("stored tables differ from canonical raw-derived analysis")
    return receipt, summary


def verify_dry_run(path, freeze_hash, epuroot, package=PACKAGE):
    compatibility = recovery.verify_compatibility(package, epuroot, path)
    verify_completed_run(path, recovery.PREDECESSOR_FREEZE, expected_phase="dry-run")
    return compatibility


@contextmanager
def attempt_ledgers(epuroot, runs, freeze):
    """Lock all retained QA1 revisions so installing a copy cannot reset budget."""
    if freeze.get("shared_attempt_directory"):
        relative = [freeze["shared_attempt_directory"],
                    *freeze.get("predecessor_attempt_directories", [])]
        directories = [diagnostics.contained_path(Path(epuroot).absolute(),
                       Path(epuroot).absolute() / name) for name in relative]
    else:
        directories = [runs]
    with ExitStack() as stack:
        ledgers = []
        for directory in sorted(set(directories)):
            directory.mkdir(parents=True, exist_ok=True)
            lock_path = directory / "ATTEMPT.lock"
            ledger = directory / "ATTEMPTS.jsonl"
            if lock_path.is_symlink() or ledger.is_symlink():
                raise ValueError("attempt lock and ledger must not be symlinks")
            lock = stack.enter_context(lock_path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledgers.append(ledger)
        yield ledgers


def run(phase, dataset, epuroot, run_id, dry_run_receipt=None, package=PACKAGE):
    if phase not in {"preflight", "development"}:
        raise ValueError("R1 authorizes preflight and one full development attempt; no new dry run")
    if phase != "preflight" and Path(package).absolute() != recovery.contained(epuroot, recovery.NATIVE_PACKAGE):
        raise ValueError("R1 execution requires its exact admitted native installation")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("invalid run ID")
    runs = package / "runs"
    runs.mkdir(exist_ok=True)
    output = runs / run_id
    output.mkdir(exist_ok=False)
    receipt = {"version": "EPU-QA1", "engineering_revision": "E5-R1",
               "recovery_version": recovery.RECOVERY_VERSION,
               "started_at": now(), "phase": phase, "run_id": run_id,
               "status": "STARTED", "data_payload_opened": False, "protected_outcomes_opened": False,
               "environment": {"python": platform.python_version(), "platform": platform.platform()},
               "claim_boundary": "Exposed-data development only; no efficacy claim without a completed run."}
    dump_new(output / "STARTED.json", receipt)
    started = time.monotonic()
    code = 2
    try:
        freeze_hash, freeze = check_freeze(package)
        receipt["freeze_sha256"] = freeze_hash
        reference = diagnostics.load_reference(epuroot)
        receipt["native_source_sha256"] = diagnostics.REFERENCE_SHA256
        if phase == "preflight":
            receipt.update(status="PASS_METADATA_ONLY", execution_admitted=False)
            code = 0
        else:
            require_resources(receipt)
            if phase == "development":
                if dry_run_receipt is None:
                    raise ValueError("development requires --dry-run-receipt")
                receipt["dry_compatibility"] = verify_dry_run(dry_run_receipt, freeze_hash, epuroot, package)
            # Hold this lock throughout data opening, computation and completion.
            with recovery.locked_recovery_ledger(package, epuroot, freeze) as ledger:
                receipt["attempt_budget_ledgers"] = [str(ledger)]
                receipt["historical_ledger_sha256"] = recovery.verify_historical_ledgers(epuroot)
                receipt["recovery_authority_before_admission"] = recovery.verify_authority(freeze_hash)
                admission = recovery.sustained_admission()
                receipt["recovery_admission"] = admission
                if admission["status"] != "PASS":
                    raise ResourceBlocked("R1 thirty-second 1 GiB admission failed")
                require_time(started)
                if check_freeze(package)[0] != freeze_hash:
                    raise ValueError("recovery freeze changed during admission")
                recovery.verify_historical_ledgers(epuroot)
                if recovery.read_recovery_attempt(package, epuroot, freeze) is not None:
                    raise ValueError("single recovery data-opening attempt already spent")
                receipt["recovery_authority_before_reservation"] = recovery.verify_authority(freeze_hash)
                attempt = {"at": now(), "phase": phase, "run_id": run_id,
                           "freeze_sha256": freeze_hash, "status": "DATA_OPENING_ATTEMPT",
                           "recovery_version": recovery.RECOVERY_VERSION}
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(attempt) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                require_time(started)
                receipt["data_payload_opened"] = True
                rows = load_digits_file(dataset)
                receipt["data_sha256"] = DATA_SHA256
                train, queries, aliases = grouped_split(rows)
                if {r["label"] for r in train} != set(range(10)) or {r["label"] for r in queries} != set(range(10)):
                    raise ValueError("training and full query splits must each cover all classes")
                if phase == "dry-run":
                    queries = queries[:32]
                noises = [0.] if phase == "dry-run" else [0., .25, .5]
                receipt.update(query_ids=[r["id"] for r in queries], noises=noises,
                               training_rows=len(train), query_groups=len(queries))
                dump_new(output / "SPLIT.json", {"training_ids": [r["id"] for r in train],
                         "query_ids": receipt["query_ids"], "all_query_aliases": aliases,
                         "role": "historically_exposed_development", "seed_policy": "PROTOCOL.md"})
                base = fit_class_means([[v / 16 for v in r["pixels"]] for r in train], [r["label"] for r in train])
                base.save(output / "MODEL.json")
                dump_new(output / "BOUNDS.json", base.bounds())
                models = {"anchor8": base,
                          "anchor1": PrototypeModel(base.memories, base.labels, Config(steps=1)),
                          "anchor_alpha0": PrototypeModel(base.memories, base.labels, Config(alpha=0.))}
                knn_started = time.perf_counter_ns()
                knn_bank = NeighborBank([unit_center([v / 16 for v in r["pixels"]]) for r in train],
                                        [r["label"] for r in train], [r["row_index"] for r in train])
                receipt["knn_preparation_elapsed_ns"] = time.perf_counter_ns() - knn_started
                receipt["logical_bank_bytes_float64"] = {"prototype": len(base.memories) * 64 * 8,
                                                          "knn": len(train) * 64 * 8}
                trace_queries = [{"id": r["id"], "vector": list(corrupted_query(r, 0.))} for r in queries[:32]]
                dump_new(output / "NATIVE_DIAGNOSTICS.json", diagnostics.analyze(reference, [list(m) for m in base.memories], trace_queries))
                dump_new(output / "ANCHOR_TRACES.json", {r["id"]: base.predict(r["vector"], trace=True) for r in trace_queries})
                require_time(started)
                accumulator = SummaryAccumulator(receipt["query_ids"], noises, phase=phase)
                processed = 0
                with (output / "RAW.jsonl").open("x", encoding="utf-8") as raw:
                    for noise in noises:
                        for row in queries:
                            require_time(started)
                            if processed % 25 == 0:
                                require_resources(receipt)
                                require_time(started)
                            q = corrupted_query(row, noise)
                            predictions = evaluate_query(reference, models, q, knn_bank)
                            for method in METHODS:
                                result = predictions[method]
                                record = {"query_id": row["id"], "noise": noise, "label": row["label"],
                                          "method": method, "initial_prediction": predictions["direct"]["prediction"],
                                          **{k: v for k, v in result.items() if k not in {"state", "trace", "scores"}}}
                                raw.write(json.dumps(record, allow_nan=False) + "\n")
                                accumulator.add(record)
                            raw.flush()
                            require_time(started)
                            processed += 1
                    raw.flush()
                    os.fsync(raw.fileno())
                require_resources(receipt)
                require_time(started)
                if check_freeze(package)[0] != freeze_hash:
                    raise ValueError("freeze changed during execution")
                diagnostics.load_reference(epuroot)
                summary = accumulator.finish()
                dump_new(output / "SUMMARY.json", summary)
                write_tables(output, summary)
                receipt["outputs"] = {
                    f.name: evidence.file_hash(f)
                    for f in output.iterdir()
                    if f.is_file() and f.name != "STARTED.json"
                }
                require_resources(receipt)
                require_time(started)
                receipt.update(status="COMPLETE_EXPOSED_DEVELOPMENT", raw_records=accumulator.count,
                               scientific_disposition="NOT_CONFIRMATORY")
                code = 0
    except ResourceBlocked as exc:
        receipt.update(status="BLOCKED_RESOURCE", error=str(exc))
    except (KeyboardInterrupt, SystemExit) as exc:
        receipt.update(status="INTERRUPTED", error=f"{type(exc).__name__}: execution interrupted")
        with (output / "ERROR.log").open("x", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        raise
    except Exception as exc:
        receipt.update(status="REFUSED_OR_FAILED", error=f"{type(exc).__name__}: {exc}")
        with (output / "ERROR.log").open("x", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
    finally:
        receipt["retained_files"] = {
            f.name: evidence.file_hash(f)
            for f in output.iterdir()
            if f.is_file() and f.name != "RECEIPT.json"
        }
        receipt.update(finished_at=now(),
                       max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                           * (1 if platform.system() == "Darwin" else 1024))
        elapsed = time.monotonic() - started
        receipt["elapsed_seconds"] = elapsed
        if code == 0 and elapsed > 600:
            receipt.update(status="REFUSED_OR_FAILED",
                           error="TimeoutError: 600-second attempt budget exceeded during finalization")
            code = 2
        dump_new(output / "RECEIPT.json", receipt)
    return code, receipt, output


def regenerate(receipt_path, output):
    receipt_path, output = Path(receipt_path), Path(output)
    freeze_hash, _ = check_freeze()
    receipt, summary = verify_completed_run(receipt_path, freeze_hash)
    output.mkdir(exist_ok=False)
    dump_new(output / "SUMMARY.json", summary)
    write_tables(output, summary)
    generated = {p.name: evidence.file_hash(p) for p in output.iterdir() if p.is_file()}
    for name, digest in generated.items():
        if digest != receipt["outputs"][name]:
            raise ValueError(f"regenerated output differs: {name}")
    dump_new(output / "REGENERATION.json", {
        "status": "PASS_REGENERATED_OUTPUTS", "scientific_disposition": "NOT_CONFIRMATORY",
        "source_receipt": str(receipt_path.absolute()),
        "source_receipt_sha256": evidence.file_hash(receipt_path, 1_000_000),
        "freeze_sha256": freeze_hash, "raw_sha256": receipt["outputs"]["RAW.jsonl"],
        "generated_outputs": generated,
        "scope": "Internal consistency and exact table regeneration, not authenticated or independent execution."})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "development"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epu-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run-receipt", type=Path)
    args = parser.parse_args()
    code, receipt, output = run(args.phase, args.dataset, args.epu_root, args.run_id, args.dry_run_receipt)
    print(json.dumps({"status": receipt["status"], "data_payload_opened": receipt["data_payload_opened"],
                      "receipt": str(output / "RECEIPT.json")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
