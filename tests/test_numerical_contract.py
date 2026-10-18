"""Small arithmetic regressions; no dataset, scientific run or efficacy test."""
import math
import unittest

from epu_query_anchor import Config, PrototypeModel, checked_state, fit_class_means, unit_center


def legacy_normalize(row):
    """E4 arithmetic retained to test its already-valid output contract exactly."""
    row = tuple(float(x) for x in row)
    mean = math.fsum(row) / len(row)
    centered = tuple(x - mean for x in row)
    divisor = max(math.hypot(*centered), 1e-12)
    return tuple(x / divisor for x in centered)


class NumericalContractTests(unittest.TestCase):
    def test_large_nearly_constant_vector_remains_centered(self):
        row = [1_000_000., 1_000_000., math.nextafter(1_000_000., -math.inf)]
        result = unit_center(row)
        checked_state(result)
        self.assertAlmostEqual(math.fsum(result), 0., places=14)
        self.assertGreater(result[0], 0.)
        self.assertAlmostEqual(result[0], result[1], places=14)
        self.assertAlmostEqual(result[2], -2 * result[0], places=14)

    def test_pixel_domain_nearly_constant_vector_remains_centered(self):
        row = [1., 1., math.nextafter(1., 0.)]
        result = unit_center(row)
        checked_state(result)
        self.assertAlmostEqual(math.fsum(result), 0., places=14)
        # Preserve the documented normalization floor for a sub-floor signal.
        self.assertLess(math.hypot(*result), 1.)
        self.assertGreater(math.hypot(*result), 0.)

    def test_class_mean_training_accepts_valid_nearly_constant_pixels(self):
        fitted = fit_class_means([[1., 1., math.nextafter(1., 0.)], [0., 1., 0.]], [0, 1])
        for memory in fitted.memories:
            checked_state(memory)
        self.assertEqual(fitted.labels, (0, 1))

    def test_valid_legacy_normalization_outputs_are_bitwise_equal(self):
        cases = [[0., 0.], [1., 1., 1.], [1., -1.], [1., 2., 0., 0.],
                 [.75, 0., .25], [1e-15, -1e-15], [1e6, -1e6, 0.],
                 [float(i % 17) / 16 for i in range(64)],
                 [(-1.) ** i * (i % 3) for i in range(128)]]
        for row in cases:
            with self.subTest(dim=len(row), row=row[:4]):
                before = legacy_normalize(row)
                checked_state(before)
                self.assertEqual(tuple(x.hex() for x in unit_center(row)),
                                 tuple(x.hex() for x in before))

    def test_beta_below_numerical_domain_is_rejected(self):
        for beta in [math.ulp(0.), 1e-300, math.nextafter(1e-12, 0.)]:
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                Config(beta=beta)

    def test_beta_boundary_energy_is_finite(self):
        for beta in [1e-12, math.nextafter(1e-12, math.inf), 6., 100.]:
            for count in [1, 2, 32]:
                with self.subTest(beta=beta, count=count):
                    bank = [[.5, -.5] if i % 2 == 0 else [-.5, .5] for i in range(count)]
                    model = PrototypeModel(bank, list(range(count)), Config(beta=beta))
                    self.assertTrue(math.isfinite(model.energy([.5, -.5], [.5, -.5])))

    def test_constant_vectors_and_default_configuration_preserved(self):
        self.assertEqual(Config().beta, 6.)
        for value in [0., 1., -1e6, 1e6]:
            for dimension in [2, 3, 64, 128]:
                self.assertEqual(unit_center([value] * dimension), (0.,) * dimension)


if __name__ == '__main__':
    unittest.main()
