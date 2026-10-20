"""Unit tests for prespecified paired continuous-outcome inference."""
import unittest
from reviewer_revision.w1_paired_stats import paired_inference, holm_adjust


class PairedStatsTests(unittest.TestCase):
    def test_all_zero_differences(self):
        result = paired_inference([0] * 5, [0] * 5, n_bootstrap=100)
        self.assertEqual(result["two_sided_sign_flip_p"], 1.0)
        self.assertEqual(result["percentile_bootstrap_95ci"], [0.0, 0.0])

    def test_all_positive_exact(self):
        result = paired_inference([2] * 5, [1] * 5, n_bootstrap=100)
        self.assertAlmostEqual(result["two_sided_sign_flip_p"], 2 / 32)

    def test_reproducible(self):
        a = list(range(25))
        b = [0] * 25
        x = paired_inference(a, b, n_permutations=99, n_bootstrap=100)
        y = paired_inference(a, b, n_permutations=99, n_bootstrap=100)
        self.assertEqual(x, y)

    def test_reject_missing(self):
        with self.assertRaises(ValueError):
            paired_inference([1.0, float("nan")], [1.0, 2.0])

    def test_holm(self):
        self.assertEqual(holm_adjust([.03, .01, .2]), [.06, .03, .2])


if __name__ == "__main__":
    unittest.main()
