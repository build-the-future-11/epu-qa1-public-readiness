"""Generated, data-free regression checks for the public hardening changes."""
from pathlib import Path
import tempfile
import unittest

from auditor import trace_audit
import epu_diagnostics as diagnostics
import epu_query_anchor as anchor
import public_preflight


class StrictAlignmentTests(unittest.TestCase):
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
