"""One explicitly versioned, nonadaptive E5 resource-recovery attempt.

Metadata and budget checks only. This module never opens the digits dataset.
"""
from contextlib import ExitStack, contextmanager
import ast
import fcntl
import hashlib
from pathlib import Path
import re
import time

import epu_diagnostics as diagnostics

RECOVERY_VERSION = "EPU-QA1-E5-R1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PREDECESSOR_PACKAGE = "successors/query_anchor_v1_e5"
NATIVE_PACKAGE = "successors/query_anchor_v1_e5_r1"
PREDECESSOR_FREEZE = "8949ebbda6cc1e991764a6fcb9cddb4b480a8b21c3e69f1eb7fdebc500059b06"
PREDECESSOR_LEDGER_SHA = "2d727f4ddfa5a74ba98d062b93b72e486de2eb576178991411dffa92c6a12b31"
PREDECESSOR_LEDGERS = ("successors/query_anchor_v1/runs/ATTEMPTS.jsonl",
                       "successors/query_anchor_v1_1/runs/ATTEMPTS.jsonl")
DRY_RUN_ID = "dry-e5-20261017T050644743799Z"
FAILED_RUN_ID = "development-e5-20261017T050644743799Z"
DRY_RECEIPT_SHA = "b6be3232ecf25f198bcb9b25a44f6411860efbb1988603ce7444b7ddb5c32a9f"
FAILED_RECEIPT_SHA = "03d3111da99e2fc1af802eaa74032b1b9430270e8af95a0ad1f073ab0e77ccbc"
SCIENTIFIC_FILES = ("epu_query_anchor.py", "epu_diagnostics.py", "safe_io.py",
                    "result_contract.py", "analyze_results.py",
                    "SCIENTIFIC_SPEC.md", "PROTOCOL.md")
SCIENTIFIC_FUNCTIONS = ("load_digits_file", "grouped_split", "corrupted_query",
                        "native_prediction", "evaluate_query", "table_bytes",
                        "write_tables", "require_time", "require_resources",
                        "verify_completed_run", "regenerate")
RECOVERY_BINDING = {
    "version": RECOVERY_VERSION,
    "canonical_state": str(REPOSITORY_ROOT / "state/projects/PRJ-71CFF039B975/RESEARCH_STATE.json"),
    "native_package": NATIVE_PACKAGE,
    "predecessor_package": PREDECESSOR_PACKAGE,
    "predecessor_freeze_sha256": PREDECESSOR_FREEZE,
    "predecessor_ledger_sha256": PREDECESSOR_LEDGER_SHA,
    "predecessor_ledgers": list(PREDECESSOR_LEDGERS),
    "dry_run_id": DRY_RUN_ID,
    "dry_receipt_sha256": DRY_RECEIPT_SHA,
    "failed_run_id": FAILED_RUN_ID,
    "failed_receipt_sha256": FAILED_RECEIPT_SHA,
    "full_data_opening_attempts": 1,
    "new_dry_data_opening_attempts": 0,
    "resource_floor_bytes": 1024**3,
    "admission_seconds": 30,
    "runtime_resource_floor_bytes": 512 * 1024**2,
    "wall_seconds": 600,
}


def read(path, limit=1_000_000):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"invalid or oversized recovery metadata: {path}")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("recovery metadata grew beyond limit")
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_freeze(freeze):
    if (freeze.get("engineering_revision") != "E5-R1"
            or freeze.get("recovery") != RECOVERY_BINDING
            or freeze.get("shared_attempt_directory") != NATIVE_PACKAGE + "/runs"
            or freeze.get("predecessor_attempt_directories") != []):
        raise ValueError("unadmitted recovery binding or budget")


def verify_authority(freeze_sha256):
    """Recheck current, explicit canonical authority immediately before spending."""
    raw = read(Path(RECOVERY_BINDING["canonical_state"]))
    state = diagnostics.strict_json(raw)
    versions = state.get("versions", [])
    if (state.get("active_study_version_id") != "EPU-QA1"
            or not isinstance(versions, list)):
        raise ValueError("R1 canonical scientific authority changed")
    active = [v for v in versions if isinstance(v, dict) and v.get("version_id") == "EPU-QA1"]
    if len(active) != 1:
        raise ValueError("R1 canonical active version ambiguous")
    authority = active[0].get("resource_recovery", {})
    if (not isinstance(authority, dict) or authority.get("recovery_id") != "E5-R1"
            or authority.get("freeze_sha256") != freeze_sha256
            or authority.get("status") not in {"ACTIVE_RESOURCE_GATED", "BLOCKED_RESOURCE"}
            or not isinstance(authority.get("budget"), dict)
            or type(authority["budget"].get("full_replacement_attempts")) is not int
            or authority["budget"]["full_replacement_attempts"] != 1):
        raise ValueError("R1 canonical execution authority unavailable or changed")
    return {"status": "PASS_CURRENT_AUTHORITY", "canonical_state_sha256": sha(raw),
            "recovery_id": "E5-R1", "freeze_sha256": freeze_sha256}


def contained(root, relative):
    return diagnostics.contained_path(Path(root).absolute(), Path(root).absolute() / relative)


def verify_historical_ledgers(epuroot):
    snapshots = {}
    for relative in PREDECESSOR_LEDGERS:
        raw = read(contained(epuroot, relative))
        if sha(raw) != PREDECESSOR_LEDGER_SHA:
            raise ValueError("historical attempt ledger identity changed")
        rows = [diagnostics.strict_json(line) for line in raw.splitlines()]
        expected = {(DRY_RUN_ID, "dry-run"), (FAILED_RUN_ID, "development")}
        if (len(rows) != 2 or any(not isinstance(row, dict) for row in rows)
                or {(row.get("run_id"), row.get("phase")) for row in rows} != expected
                or any(row.get("freeze_sha256") != PREDECESSOR_FREEZE
                       or row.get("status") != "DATA_OPENING_ATTEMPT" for row in rows)):
            raise ValueError("historical attempt ledger does not match recovery decision")
        snapshots[str(contained(epuroot, relative))] = sha(raw)
    return snapshots


def read_recovery_attempt(package, epuroot, freeze):
    """Read-only budget inspection; returns None or the sole reserved attempt.

    Any nonempty invalid/truncated ledger fails closed. Callers may inspect this
    before resource admission; do not use its result instead of locked admission.
    """
    validate_freeze(freeze)
    verify_historical_ledgers(epuroot)
    ledger = contained(epuroot, NATIVE_PACKAGE + "/runs/ATTEMPTS.jsonl")
    if ledger.is_symlink():
        raise ValueError("recovery ledger must not be a symlink")
    if not ledger.exists():
        return None
    raw = read(ledger)
    if not raw:
        return None
    if not raw.endswith(b"\n") or len(raw.splitlines()) != 1:
        raise ValueError("malformed recovery ledger; budget remains unavailable")
    row = diagnostics.strict_json(raw)
    expected_keys = {"at", "phase", "run_id", "freeze_sha256", "status", "recovery_version"}
    current_freeze = sha(read(Path(package) / "FREEZE.json", 100_000))
    if (not isinstance(row, dict) or set(row) != expected_keys
            or row.get("phase") != "development"
            or row.get("status") != "DATA_OPENING_ATTEMPT"
            or row.get("recovery_version") != RECOVERY_VERSION
            or row.get("freeze_sha256") != current_freeze
            or not isinstance(row.get("at"), str) or not row["at"]
            or not isinstance(row.get("run_id"), str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", row["run_id"])):
        raise ValueError("invalid recovery ledger; budget remains unavailable")
    return row


@contextmanager
def locked_recovery_ledger(package, epuroot, freeze):
    validate_freeze(freeze)
    package = Path(package).absolute()
    directories = [contained(epuroot, relative).parent for relative in PREDECESSOR_LEDGERS]
    runs = contained(epuroot, NATIVE_PACKAGE + "/runs")
    if runs.is_symlink():
        raise ValueError("recovery runs directory must not be a symlink")
    runs.mkdir(exist_ok=True)
    directories.append(runs)
    with ExitStack() as stack:
        for directory in sorted(set(directories)):
            lock = directory / "ATTEMPT.lock"
            if lock.is_symlink():
                raise ValueError("attempt lock must not be a symlink")
            handle = stack.enter_context(lock.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_historical_ledgers(epuroot)
        if read_recovery_attempt(package, epuroot, freeze) is not None:
            raise ValueError("single recovery data-opening attempt already spent; no retry authorized")
        yield runs / "ATTEMPTS.jsonl"
        verify_historical_ledgers(epuroot)


def verify_compatibility(package, epuroot, dry_receipt):
    """Bind old dry evidence to identical scientific computation, without rerun."""
    old = contained(epuroot, PREDECESSOR_PACKAGE)
    raw = read(old / "FREEZE.json", 100_000)
    if sha(raw) != PREDECESSOR_FREEZE:
        raise ValueError("predecessor freeze changed")
    previous = diagnostics.strict_json(raw)
    for name, digest in previous["files"].items():
        if not isinstance(name, str) or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("invalid predecessor file binding")
        if sha(read(old / name)) != digest:
            raise ValueError("predecessor frozen source changed")
    for name in SCIENTIFIC_FILES:
        if sha(read(Path(package) / name)) != previous["files"][name]:
            raise ValueError(f"scientific source changed: {name}")
    old_runner, new_runner = read(old / "run_study.py"), read(Path(package) / "run_study.py")
    def functions(raw):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}
    old_functions, new_functions = functions(old_runner), functions(new_runner)
    if any(old_functions[name] != new_functions[name] for name in SCIENTIFIC_FUNCTIONS):
        raise ValueError("scientific runner function changed")
    # The full data load, model preparation, fixed matrix loop, statistics and
    # finalization are copied verbatim; only admission/reservation is amended.
    def execution_body(raw):
        text = raw.decode("utf-8")
        start = text.index("                rows = load_digits_file(dataset)")
        return text[start:text.index("    except ResourceBlocked as exc:", start)]
    if execution_body(old_runner) != execution_body(new_runner):
        raise ValueError("scientific execution body changed")
    expected_dry = contained(epuroot, PREDECESSOR_PACKAGE + "/runs/" + DRY_RUN_ID + "/RECEIPT.json")
    if Path(dry_receipt).absolute() != expected_dry:
        raise ValueError("only the pinned completed E5 dry receipt is compatible")
    if sha(read(expected_dry)) != DRY_RECEIPT_SHA:
        raise ValueError("completed E5 dry receipt changed")
    failed_path = contained(epuroot, PREDECESSOR_PACKAGE + "/runs/" + FAILED_RUN_ID + "/RECEIPT.json")
    failed_raw = read(failed_path)
    if sha(failed_raw) != FAILED_RECEIPT_SHA:
        raise ValueError("failed E5 receipt changed")
    failed = diagnostics.strict_json(failed_raw)
    if (failed.get("status") != "BLOCKED_RESOURCE" or failed.get("data_payload_opened") is not True
            or failed.get("phase") != "development" or failed.get("freeze_sha256") != PREDECESSOR_FREEZE):
        raise ValueError("recovery cause does not match preserved failure")
    return {"status": "PASS_IDENTICAL_SCIENTIFIC_COMPUTATION",
            "predecessor_freeze_sha256": PREDECESSOR_FREEZE,
            "dry_receipt_sha256": DRY_RECEIPT_SHA,
            "failed_receipt_sha256": FAILED_RECEIPT_SHA,
            "scope": "Same scientific source and fixed execution body; dry bundle must also validate."}


def sustained_admission(sample=diagnostics.resource_sample, pause=time.sleep, clock=time.monotonic):
    """Require 31 passing samples separated by >=1s and spanning >=30s.

    Sampling cannot establish headroom between observations. A refusal returns
    immediately; no data-opening budget is reserved by this function.
    """
    result = {"status": "BLOCKED_RESOURCE", "floor_bytes": 1024**3,
              "minimum_span_seconds": 30, "minimum_samples": 31,
              "samples": [], "scope": "Sampled sustained headroom; not continuous monitoring."}
    started = clock()
    previous = None
    try:
        for index in range(31):
            if index:
                pause(1.0)
            current = sample()
            elapsed = clock() - started
            fields = ("free_plus_speculative_bytes", "pageouts", "swap_used_bytes")
            if (not isinstance(current, dict) or any(type(current.get(k)) is not int
                    or current[k] < 0 for k in fields)):
                raise ValueError("invalid resource sample")
            result["samples"].append({"elapsed_seconds": elapsed, **current})
            if (current["free_plus_speculative_bytes"] < 1024**3
                    or (previous is not None and (current["pageouts"] != previous["pageouts"]
                        or current["swap_used_bytes"] > previous["swap_used_bytes"]))):
                return result
            if previous is not None and elapsed - result["samples"][-2]["elapsed_seconds"] < 1.0:
                raise ValueError("resource sampling interval shorter than one second")
            previous = current
        if result["samples"][-1]["elapsed_seconds"] - result["samples"][0]["elapsed_seconds"] < 30.0:
            raise ValueError("resource observation span shorter than thirty seconds")
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
