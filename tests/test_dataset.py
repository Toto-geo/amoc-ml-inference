"""Focused checks for the dataset rules: blocked folds, train-only stats, seasonal encoding."""

import unittest

import numpy as np

from machine_learning.dataset import fold_indices, normalize, season


class DatasetRulesTest(unittest.TestCase):
    def test_blocked_fold_roles(self):
        for length in (816, 272):
            size = length // 8
            for fold in range(1, 9):
                roles = fold_indices(length, fold)
                self.assertEqual([len(roles[k]) for k in ("train", "val", "test")],
                                 [6 * size, size, size])
                self.assertEqual(len(np.unique(np.concatenate(list(roles.values())))), length)
                self.assertEqual(roles["test"][0], (fold - 1) * size)

    def test_normalization_ignores_held_out_values(self):
        values = np.array([[1.0], [3.0], [5.0], [1000.0]], dtype=np.float32)
        normalized, mean, std = normalize(values, np.array([0, 1, 2]))
        self.assertEqual(float(mean[0]), 3.0)
        self.assertAlmostEqual(float(std[0]), np.sqrt(8.0 / 3.0))
        self.assertAlmostEqual(float(normalized[1, 0]), 0.0)

    def test_quarter_season_uses_quarter_not_month(self):
        dates = np.array(["1958-01", "1958-03", "1958-04", "1958-07"], dtype="datetime64[M]")
        encoded = season(dates, "Q")
        np.testing.assert_array_equal(encoded[0], encoded[1])
        np.testing.assert_allclose(encoded[0], [0.0, 1.0], atol=1e-7)
        np.testing.assert_allclose(encoded[2], [1.0, 0.0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
