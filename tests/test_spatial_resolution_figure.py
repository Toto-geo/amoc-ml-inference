"""Tests for the spatial-resolution figure generator (fig09): fold-spread and summary-reading paths."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from figures.scripts.fig09_spatial_resolution import BINS, REGIMES, load_runs, plot_data
from machine_learning.train import SEEDS


class SpatialResolutionFigureTests(unittest.TestCase):
    def test_seed_means_then_fold_spread(self):
        rows = [{"regime": regime, "surface_n_bins": bins, "fold": fold,
                 "seed": seed, "test_rmse_phys": 4 + fold / 10 + seed / 100000,
                 "test_pearson": 0.5 + fold / 100,
                 "test_spread_ratio": 0.7 + fold / 100}
                for regime in REGIMES for bins in BINS
                for fold in range(1, 9) for seed in SEEDS]
        data = plot_data(pd.DataFrame(rows))
        self.assertEqual(len(data), 24)
        rmse = data.loc[data.regime.eq("M") & data.surface_n_bins.eq(20)
                        & data.metric.eq("test_rmse_phys")].iloc[0]
        self.assertAlmostEqual(rmse["fold_std"], np.std(np.arange(1, 9) / 10, ddof=1))
        with self.assertRaises(ValueError):
            plot_data(pd.DataFrame(rows[:-1]))

    def test_reads_new_trainer_summaries(self):
        with tempfile.TemporaryDirectory() as temp, patch("figures.scripts.fig09_spatial_resolution.SEEDS", (7,)):
            root = Path(temp)
            for regime in REGIMES:
                for bins in BINS:
                    for fold in range(1, 9):
                        path = root / f"{regime}_{bins}" / f"fold_{fold:02d}" / "seed_00007"
                        path.mkdir(parents=True)
                        (path / "summary.json").write_text(json.dumps({
                            "config": {"regime": regime, "bins": bins, "fold": fold, "seed": 7},
                            "test_used_for_model_selection": False,
                            "test_metrics": {"rmse_phys": 4.0, "pearson": 0.5,
                                             "spread_ratio": 0.8},
                        }))
            self.assertEqual(len(load_runs(root)), 64)

    def test_reads_historical_summaries(self):
        with tempfile.TemporaryDirectory() as temp, patch("figures.scripts.fig09_spatial_resolution.SEEDS", (7,)):
            root = Path(temp)
            for regime in REGIMES:
                for bins in BINS:
                    for fold in range(1, 9):
                        path = root / f"{regime}_{bins}" / f"fold_{fold:02d}" / "seed_00007"
                        path.mkdir(parents=True)
                        (path / "summary.json").write_text(json.dumps({
                            "regime": regime, "fold": fold, "seed": 7,
                            "dataset_model_contract": {"surface_n_bins": bins},
                            "test_used_for_model_selection": False,
                            "test_metrics": {"test_rmse_phys": 4.0,
                                             "test_pearson": 0.5,
                                             "test_spread_ratio": 0.8},
                        }))
            self.assertEqual(len(load_runs(root)), 64)


if __name__ == "__main__":
    unittest.main()
