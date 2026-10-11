"""Data-free contract checks for the public PaperModel interface and checkpoint loader."""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from machine_learning.model import build_model, compute_initial_centers, load_checkpoint


class PaperModelTest(unittest.TestCase):
    def test_historical_checkpoint_without_latitude_rows(self):
        surfaces = {name: np.array([1.0]) for name in ("sst", "sss", "ssh", "obp")}
        argo = {"rho": np.array([1.0])}
        stats = {
            "locked_hpo_trial": "HPO1_M/trial_0340",
            "y_mean_train": 10.0, "y_std_train": 5.0,
            "argo_norm_means": argo, "argo_norm_stds": argo,
            "surface_norm_means": surfaces, "surface_norm_stds": surfaces,
        }
        contract = {"regime": "M", "fold": 5, "surface_n_bins": 20,
                    "surface_lat_rows": 20, "surface_temporal_mode": "monthly_1"}
        checkpoint = {"config": {**contract, "surface_include_ws": False},
                      "norm_stats": stats, "state_dict": {}}
        model = SimpleNamespace(dataset_contract=contract, normalization=stats,
                                load_state_dict=Mock())
        del checkpoint["config"]["surface_lat_rows"]
        with patch("machine_learning.model.torch.load", return_value=checkpoint):
            self.assertIs(load_checkpoint(model, "historical.pt"), checkpoint)
            model.load_state_dict.assert_called_once_with({}, strict=True)
            checkpoint["config"]["surface_lat_rows"] = 5
            with self.assertRaisesRegex(ValueError, "surface_lat_rows"):
                load_checkpoint(model, "wrong_rows.pt")
            del checkpoint["config"]["surface_lat_rows"]
            model.dataset_contract = {**contract, "surface_lat_rows": 5}
            with self.assertRaisesRegex(ValueError, "surface_lat_rows"):
                load_checkpoint(model, "wrong_legacy_grid.pt")

    def test_kmeans_centers_are_recomputed_from_training_coordinates(self):
        coordinates = np.linspace(0, 1, 52, dtype=np.float32).reshape(2, 26)
        train = SimpleNamespace(lon_scaled=coordinates, lat_scaled=coordinates / 2,
                                days_scaled=np.full_like(coordinates, 0.5))
        first = compute_initial_centers(train)
        second = compute_initial_centers(train)
        self.assertEqual(first.shape, (26, 3))
        np.testing.assert_array_equal(first, second)
        train.F = 54
        train.norm_stats = {
            "regime": "M", "fold_number": 1, "surface_n_bins": 20,
            "surface_lat_rows": 20, "surface_temporal_mode": "monthly_1",
            "surface_feature_dims": {name: 20 for name in ("sst", "sss", "ssh", "obp")},
            "surface_months_per_sample": 1, "season_feature_dim": 2,
        }
        with patch("machine_learning.model.np.load", side_effect=AssertionError("NPZ accessed")):
            model = build_model(train)
        np.testing.assert_allclose(model.node_assigner.centers.detach().numpy(), first,
                                   rtol=0, atol=1e-5)

    def test_reference_centers_cover_every_fold(self):
        path = Path(__file__).resolve().parents[1] / "machine_learning/initial_centers.npz"
        with np.load(path) as reference:
            self.assertEqual(set(reference.files), {
                f"{regime}_{fold:02d}" for regime in ("M", "Q") for fold in range(1, 9)
            })
            for values in reference.values():
                self.assertEqual(values.shape, (26, 3))
                self.assertEqual(values.dtype, np.float32)
                self.assertTrue(np.isfinite(values).all())

    def test_four_surface_interface(self):
        for regime, bins, rows, mode in (("M", 20, 20, "monthly_1"),
                                        ("Q", 160, 3, "quarter_mean")):
            with self.subTest(regime=regime, bins=bins):
                train = SimpleNamespace(F=54, norm_stats={
                    "regime": regime, "fold_number": 1,
                    "surface_n_bins": bins, "surface_lat_rows": rows,
                    "surface_temporal_mode": mode,
                    "surface_feature_dims": {name: bins for name in ("sst", "sss", "ssh", "obp")},
                    "surface_months_per_sample": 1, "season_feature_dim": 2,
                })
                model = build_model(train, initialize_assignment=False).eval()
                batch = {
                    "x": torch.zeros(2, 3, 54), "mask_padded": torch.ones(2, 3),
                    "lon": torch.full((2, 3), 0.5), "lat": torch.full((2, 3), 0.5),
                    "days": torch.full((2, 3), 0.5), "season": torch.zeros(2, 2),
                    **{name: torch.zeros(2, bins) for name in ("sst", "sss", "ssh", "obp")},
                }
                with torch.inference_mode():
                    prediction, _ = model(batch)
                self.assertEqual(tuple(prediction.shape), (2,))
                self.assertTrue(bool(torch.isfinite(prediction).all()))
                self.assertNotIn("ws", batch)
                self.assertEqual(model.surface_ablate_vars, ("ws",))


if __name__ == "__main__":
    unittest.main()
