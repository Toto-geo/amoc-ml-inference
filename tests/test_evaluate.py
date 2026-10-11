"""Tests for the prediction-first aggregation contract (seed mean then fold skill)."""

import unittest

import numpy as np
import pandas as pd

from machine_learning.dataset import fold_indices
from machine_learning.evaluate import aggregate_predictions, skill


def example_predictions() -> pd.DataFrame:
    rows = []
    for fold in (1, 2):
        for index in fold_indices(816, fold)["test"]:
            truth = 10 + np.sin(index / 7) + index / 1000
            for seed, offset in ((42, -3.0), (123, 3.0)):
                rows.append({"regime": "M", "bins": 20, "fold": fold,
                             "seed": seed, "global_time_index": int(index),
                             "date": str(np.datetime64("1958-01") + np.timedelta64(int(index), "M")),
                             "y_true_phys": truth,
                             "y_pred_phys": truth * 0.8 + offset})
    return pd.DataFrame(rows)


class EvaluationTests(unittest.TestCase):
    def test_seed_mean_precedes_fold_metrics(self):
        mean, folds, summary = aggregate_predictions(example_predictions(),
                                                     expected_folds=2, expected_seeds=2)
        self.assertEqual(len(mean), 204)
        self.assertEqual(len(folds), 2)
        self.assertEqual(int(summary.iloc[0].bins), 20)
        for fold in (1, 2):
            part = mean[mean.fold.eq(fold)]
            expected = skill(part.y_true_phys.to_numpy(), part.y_pred_phys.to_numpy())
            actual = folds[folds.fold.eq(fold)].iloc[0]
            for metric, value in expected.items():
                self.assertAlmostEqual(actual[metric], value, places=12)
        raw_rmse = np.mean([skill(part.y_true_phys.to_numpy(), part.y_pred_phys.to_numpy())["rmse"]
                            for _, part in example_predictions().groupby(["fold", "seed"])])
        self.assertNotAlmostEqual(float(folds.rmse.mean()), raw_rmse, places=3)

    def test_missing_sample_or_disagreeing_truth_is_rejected(self):
        frame = example_predictions()
        with self.assertRaisesRegex(ValueError, "Incomplete test block"):
            aggregate_predictions(frame.iloc[1:], expected_folds=2, expected_seeds=2)
        frame.loc[0, "y_true_phys"] += 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            aggregate_predictions(frame, expected_folds=2, expected_seeds=2)


if __name__ == "__main__":
    unittest.main()
