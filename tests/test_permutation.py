"""Tests for the Argo-source and joint-surface-bin permutation routines."""

import unittest

import pandas as pd
import torch

from machine_learning.analysis.permutation import permute, permutation_indices, permutation_seed, summarize


class PermutationTests(unittest.TestCase):
    def setUp(self):
        self.batch = {key: torch.arange(12, dtype=torch.float32).reshape(3, 4) + offset
                      for offset, key in enumerate(("x", "mask_padded", "lon", "lat", "days",
                                                     "sst", "sss", "ssh", "obp", "season", "y"))}
        self.indices = torch.tensor([2, 0, 1])

    def test_historical_seed_schedule(self):
        self.assertEqual(permutation_seed(90210, "exp1_M_full", 5, 42, "argo", 0), 575502344)
        self.assertEqual(permutation_seed(314159, "exp1_M_full", 5, 42,
                                          "surface_bin", 1, 0), 343876581)
        self.assertTrue(torch.equal(permutation_indices(8, 575502344),
                                    permutation_indices(8, 575502344)))

    def test_argo_permutation_keeps_profiles_together(self):
        changed = permute(self.batch, self.indices, source="argo")
        for key in ("x", "mask_padded", "lon", "lat", "days"):
            torch.testing.assert_close(changed[key], self.batch[key][self.indices])
        for key in ("sst", "sss", "ssh", "obp", "season", "y"):
            torch.testing.assert_close(changed[key], self.batch[key])

    def test_bin_permutation_moves_four_fields_jointly(self):
        changed = permute(self.batch, self.indices, bin_index=2)
        for key in ("sst", "sss", "ssh", "obp"):
            torch.testing.assert_close(changed[key][:, 1], self.batch[key][self.indices, 1])
            torch.testing.assert_close(changed[key][:, [0, 2, 3]], self.batch[key][:, [0, 2, 3]])
        torch.testing.assert_close(changed["x"], self.batch["x"])
        with self.assertRaises(ValueError):
            permute(self.batch, self.indices, bin_index=5)

    def test_summary_rejects_missing_repeats(self):
        rows = [{"regime": "M", "bins": 20, "kind": "source", "source": "argo",
                 "bin_index": 0, "fold": fold, "seed": seed, "repeat": repeat,
                 "mae_increase_pct": fold + repeat}
                for fold in (1, 2) for seed in (42, 123) for repeat in (0, 1)]
        frame = pd.DataFrame(rows)
        folds, summary = summarize(frame, expected_folds=2, expected_seeds=2, expected_repeats=2)
        self.assertEqual(len(folds), 2)
        self.assertAlmostEqual(summary.iloc[0].mean_pct, 2.0)
        with self.assertRaisesRegex(ValueError, "Incomplete permutation repeats"):
            summarize(frame.iloc[1:], expected_folds=2, expected_seeds=2, expected_repeats=2)


if __name__ == "__main__":
    unittest.main()
