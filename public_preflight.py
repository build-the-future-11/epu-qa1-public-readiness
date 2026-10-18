#!/usr/bin/env python3
"""Data-free public verification for the EPU-QA1 public-readiness candidate."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "candidate"
AUDITOR = HERE / "auditor"
TESTS = HERE / "tests"
MAX_SOURCE_BYTES = 1_000_000
GENERATED_DIRS = {".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return the fields that must remain stable for an admitted source file."""
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stable_hash(path: Path) -> str:
    """Hash one bounded regular file while refusing links and replacements."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, flags), "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"non-regular public source: {path}")
        if before.st_size > MAX_SOURCE_BYTES:
            raise ValueError(f"oversized public source: {path}")
        while block := handle.read(65_536):
            digest.update(block)
        after = os.fstat(handle.fileno())
    final = path.stat(follow_symlinks=False)
    if file_identity(before) != file_identity(after) or file_identity(after) != file_identity(final):
        raise ValueError(f"source changed while hashing: {path}")
    return digest.hexdigest()


def python_files() -> list[Path]:
    files = sorted((HERE / "public_preflight.py", *CANDIDATE.glob("*.py"),
                    *AUDITOR.glob("*.py"), *TESTS.glob("*.py")))
    if not files:
        raise ValueError("public candidate contains no Python source")
    return files


def release_files(root: Path = HERE) -> list[Path]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not path.is_relative_to(root / "verification")
        and not GENERATED_DIRS.intersection(path.relative_to(root).parts)
        and path.suffix not in {".pyc", ".pyo"}
    )
    if not files:
        raise ValueError("public release contains no source files")
    return files


def ast_checks(paths: list[Path]) -> dict[str, int]:
    zip_calls = 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name):
                if node.type.id == "BaseException":
                    raise ValueError(f"BaseException catch is forbidden: {path}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip":
                zip_calls += 1
                strict = next((keyword.value for keyword in node.keywords if keyword.arg == "strict"), None)
                if not isinstance(strict, ast.Constant) or strict.value is not True:
                    raise ValueError(f"zip must fail on length mismatch: {path}:{node.lineno}")
    return {"python_files": len(paths), "strict_zip_calls": zip_calls}


def portable_text(value: str, *, scratch: Path) -> str:
    """Remove checkout- and run-specific absolute paths from public receipts."""
    replacements = (
        (str(scratch.parent.resolve()), "<OUTPUT>"),
        (str(HERE), "<REPOSITORY>"),
    )
    for original, replacement in replacements:
        value = value.replace(original, replacement)
    return value


def validated_output(path: Path) -> Path:
    """Return a new output path strictly beneath this release's verification tree."""
    verification = (HERE / "verification").resolve()
    resolved = path.resolve(strict=False)
    if resolved == verification or not resolved.is_relative_to(verification):
        raise ValueError("--output must be a new directory beneath verification/")
    return resolved


def command(args: list[str], *, timeout: int, scratch: Path) -> dict[str, object]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join((str(CANDIDATE), str(TESTS))),
        "EPU_REFERENCE_ROOT": str(HERE / "reference"),
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": str(scratch),
    }
    completed = subprocess.run(
        args,
        cwd=HERE,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout
    if len(output.encode("utf-8")) > 1_000_000:
        raise ValueError("verification command output exceeded bound")
    recorded_args = list(args)
    if recorded_args and Path(recorded_args[0]).is_absolute():
        recorded_args[0] = Path(recorded_args[0]).name
    return {
        "argv": [portable_text(value, scratch=scratch) for value in recorded_args],
        "exit_code": completed.returncode,
        "output": portable_text(output, scratch=scratch),
    }


def write_new(path: Path, value: object) -> None:
    """Create and durably synchronize a JSON receipt and its directory entry."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = validated_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "tmp"
    scratch.mkdir()
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "STARTED",
        "started_at": now(),
        "scope": "Data-free source and generated-fixture verification; no scientific outcome.",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "data_payload_opened": False,
        "scientific_disposition": "NOT_EVALUATED",
        "checks": [],
    }
    code = 2
    try:
        paths = python_files()
        source_manifest = {
            str(path.relative_to(HERE)): stable_hash(path) for path in release_files()
        }
        receipt["source_sha256"] = source_manifest
        canonical_manifest = json.dumps(
            source_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        receipt["source_manifest_sha256"] = hashlib.sha256(canonical_manifest).hexdigest()
        core_paths = [
            path for path in paths
            if path == HERE / "public_preflight.py"
            or path.is_relative_to(CANDIDATE)
            or path.is_relative_to(AUDITOR)
        ]
        receipt["ast"] = ast_checks(core_paths)
        compile_check = command(
            [sys.executable, "-B", "-m", "compileall", "-q", "public_preflight.py",
             "candidate", "auditor", "tests"],
            timeout=60,
            scratch=scratch,
        )
        receipt["checks"].append({"name": "compileall", **compile_check})
        if compile_check["exit_code"]:
            raise ValueError("compileall failed")
        unit_check = command(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
            timeout=180,
            scratch=scratch,
        )
        receipt["checks"].append({"name": "generated_and_contract_tests", **unit_check})
        if unit_check["exit_code"]:
            raise ValueError("unit/contract tests failed")
        ruff = shutil.which("ruff")
        if ruff:
            lint_check = command(
                [ruff, "check", "public_preflight.py", "candidate", "auditor",
                 "--select", "E9,F,B,C4,PIE,RUF,PERF"],
                timeout=60,
                scratch=scratch,
            )
            receipt["checks"].append({"name": "ruff", **lint_check})
            if lint_check["exit_code"]:
                raise ValueError("ruff failed")
        else:
            receipt["checks"].append({"name": "ruff", "status": "SKIPPED_NOT_INSTALLED"})
        receipt["status"] = "PASS_PUBLIC_SOURCE_REVIEW"
        code = 0
    except (KeyboardInterrupt, SystemExit) as exc:
        error = portable_text(f"{type(exc).__name__}: {exc}", scratch=scratch)
        receipt.update(status="INTERRUPTED", error=error)
        raise
    except Exception as exc:
        error = portable_text(f"{type(exc).__name__}: {exc}", scratch=scratch)
        receipt.update(status="FAIL_PUBLIC_SOURCE_REVIEW", error=error)
    finally:
        receipt["finished_at"] = now()
        receipt_path = output / "RECEIPT.json"
        write_new(receipt_path, receipt)
        display_path = portable_text(str(receipt_path), scratch=scratch)
        print(json.dumps({"status": receipt["status"], "receipt": display_path}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
