"""Explicit unchanged E5 math/data/statistics cases; see TEST_CONTRACT.md."""
import unittest
import os
from pathlib import Path
import legacy_test_epu_query_anchor as model
import legacy_test_prerun_integrity as protocol

# The staged candidate is outside the native repository. Resolve its retained
# reference explicitly; diagnostics still verifies the pinned source hash.
model.NATIVE = Path(os.environ.get(
    "EPU_REFERENCE_ROOT", str(Path(__file__).resolve().parents[1] / "reference")
))

EXPERIMENT_CASES = (
    "test_duplicate_groups_never_cross_split",
    "test_conflicting_duplicate_label_refuses",
    "test_noise_is_reproducible_and_query_specific",
    "test_paired_summary_and_incomplete_refusal",
    "test_native_and_candidate_matrix_executes",
)
STATISTICS_CASES = (
    "test_exact_one_percentage_point_passes",
    "test_below_and_above_threshold",
    "test_dry_run_never_nominates_or_closes_utility",
    "test_unknown_phase_refused",
    "test_label_cannot_change_across_corruptions",
    "test_malformed_result_values_refused",
    "test_duplicate_expected_axes_refused",
    "test_initial_prediction_must_match_direct",
)

def load_tests(loader, tests, pattern):
    suite = loader.loadTestsFromTestCase(model.ModelTests)
    suite.addTests(model.ExperimentTests(name) for name in EXPERIMENT_CASES)
    suite.addTests(protocol.PreRunIntegrityTests(name) for name in STATISTICS_CASES)
    return suite
