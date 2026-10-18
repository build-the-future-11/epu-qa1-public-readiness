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
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "candidate"
AUDITOR = HERE / "auditor"
TESTS = HERE / "tests"
MAX_SOURCE_BYTES = 1_000_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"non-regular public source: {path}")
    before = path.stat()
    if before.st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"oversized public source: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(65_536):
            digest.update(block)
    after = path.stat()
    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if identity(before) != identity(after):
        raise ValueError(f"source changed while hashing: {path}")
    return digest.hexdigest()


def python_files() -> list[Path]:
    files = sorted((HERE / "public_preflight.py", *CANDIDATE.glob("*.py"),
                    *AUDITOR.glob("*.py"), *TESTS.glob("*.py")))
    if not files:
        raise ValueError("public candidate contains no Python source")
    return files


def release_files() -> list[Path]:
    files = sorted(
        path for path in HERE.rglob("*")
        if path.is_file()
        and not path.is_relative_to(HERE / "verification")
        and "__pycache__" not in path.parts
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
    return {"argv": args, "exit_code": completed.returncode, "output": output}


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = args.output / "tmp"
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
        receipt["source_sha256"] = {
            str(path.relative_to(HERE)): stable_hash(path) for path in release_files()
        }
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
    except Exception as exc:
        receipt.update(status="FAIL_PUBLIC_SOURCE_REVIEW", error=f"{type(exc).__name__}: {exc}")
    finally:
        receipt["finished_at"] = now()
        write_new(args.output / "RECEIPT.json", receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(args.output / "RECEIPT.json")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
