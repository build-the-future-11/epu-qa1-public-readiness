"""Generated, data-free regression checks for the public hardening changes."""
import json
from pathlib import Path
import re
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from auditor import trace_audit
import epu_diagnostics as diagnostics
import epu_query_anchor as anchor
import public_preflight


class StrictAlignmentTests(unittest.TestCase):
    def test_stable_hash_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"fixture")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises((OSError, ValueError)):
                public_preflight.stable_hash(link)

    def test_write_new_is_exclusive_and_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RECEIPT.json"
            public_preflight.write_new(path, {"status": "PASS"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "PASS"})
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            with self.assertRaises(FileExistsError):
                public_preflight.write_new(path, {"status": "REPLACED"})

    def test_preflight_records_interrupt_before_propagating(self):
        with tempfile.TemporaryDirectory(dir=public_preflight.HERE / "verification") as directory:
            output = Path(directory) / "interrupted"
            argv = ["public_preflight.py", "--output", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(public_preflight, "python_files", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                public_preflight.main()
            receipt = json.loads((output / "RECEIPT.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "INTERRUPTED")
            self.assertFalse(receipt["data_payload_opened"])

    def test_output_must_remain_beneath_verification(self):
        outside = Path("/tmp") / "epu-public-preflight-outside"
        with self.assertRaisesRegex(ValueError, "beneath verification"):
            public_preflight.validated_output(outside)

    def test_portable_text_redacts_checkout_and_output_paths(self):
        scratch = public_preflight.HERE / "verification" / "run-id" / "tmp"
        raw = f"{public_preflight.HERE}/candidate {scratch.parent}/RECEIPT.json"
        cleaned = public_preflight.portable_text(raw, scratch=scratch)
        self.assertEqual(cleaned, "<REPOSITORY>/candidate <OUTPUT>/RECEIPT.json")
        self.assertNotIn(str(public_preflight.HERE), cleaned)

    def test_release_manifest_excludes_generated_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retained = (root / ".github" / "workflows" / "verify.yml", root / ".gitignore")
            generated = (
                root / ".ruff_cache" / "entry",
                root / ".git" / "objects" / "entry",
                root / "__pycache__" / "entry.pyc",
                root / "verification" / "run" / "RECEIPT.json",
            )
            for path in (*retained, *generated):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            manifest = {path.relative_to(root).as_posix() for path in public_preflight.release_files(root)}
            self.assertTrue(all(path.relative_to(root).as_posix() not in manifest for path in generated))
            self.assertEqual(manifest, {".github/workflows/verify.yml", ".gitignore"})

    def test_release_version_metadata_agree(self):
        root = public_preflight.HERE
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        version = metadata["project"]["version"]
        citation = (root / "CITATION.cff").read_text(encoding="utf-8")
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertRegex(citation, rf"(?m)^version: {re.escape(version)}$")
        self.assertRegex(changelog, rf"(?m)^## {re.escape(version)} - \d{{4}}-\d{{2}}-\d{{2}}$")

    def test_readme_local_links_resolve(self):
        root = public_preflight.HERE
        readme = (root / "README.md").read_text(encoding="utf-8")
        local_links = [
            target for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
            if "://" not in target and not target.startswith("#")
        ]
        self.assertTrue(local_links)
        self.assertEqual([], [target for target in local_links if not (root / target).is_file()])

    def test_model_dot_refuses_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            anchor.dot((1.0, 2.0), (1.0,))

    def test_auditor_dot_refuses_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            trace_audit.dot([1.0, 2.0], [1.0])

    def test_diagnostic_covariance_refuses_unaligned_inputs(self):
        memories = [[0.5, -0.5], [-0.5, 0.5]]
        with self.assertRaises(ValueError):
            diagnostics.covariance_action(memories, [1.0], [0.0, 0.0], [0.5, -0.5])

    def test_fit_rejects_non_config_object(self):
        with self.assertRaisesRegex(ValueError, "Config required"):
            anchor.fit_class_means([[0.0, 1.0]], [0], config={})


if __name__ == "__main__":
    unittest.main()
