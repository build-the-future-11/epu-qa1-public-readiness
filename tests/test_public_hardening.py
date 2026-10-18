"""Generated, data-free regression checks for the public hardening changes."""
import unittest

from auditor import trace_audit
import epu_diagnostics as diagnostics
import epu_query_anchor as anchor


class StrictAlignmentTests(unittest.TestCase):
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
