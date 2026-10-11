"""Tests for the source and regional permutation figure generators (fig07 and fig08)."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from figures.scripts.fig08_regional_contribution import plot_data as regional_plot_data
from figures.scripts.fig07_source_contribution import build_figure, plot_data as source_plot_data


class PermutationFigureTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for regime in ("M", "Q"):
            for source in ("argo", "sst", "sss", "ssh", "obp"):
                rows.append({"regime": regime, "bins": 20, "kind": "source",
                             "source": source, "bin_index": 0,
                             "mean_pct": 30.0 if source == "argo" else 2.0})
            for index in range(1, 21):
                rows.append({"regime": regime, "bins": 20, "kind": "bin",
                             "source": "surface_block", "bin_index": index,
                             "mean_pct": float(index - 2)})
        self.summary = pd.DataFrame(rows)

    def test_source_figure_uses_five_final_sources(self):
        data = source_plot_data(self.summary)
        self.assertEqual(len(data), 10)
        fig = build_figure(data)
        try:
            self.assertEqual([text.get_text() for text in fig.axes[0].get_yticklabels()][0],
                             "Argo")
        finally:
            plt.close(fig)
        with self.assertRaises(ValueError):
            source_plot_data(self.summary.loc[~self.summary.source.eq("sst")])

    def test_regional_figure_requires_complete_geometry(self):
        edges = np.linspace(-51.5, 17.5, 21)
        dataset = xr.Dataset({"bin_edge_left": ("bin", edges[:-1]),
                              "bin_edge_right": ("bin", edges[1:]),
                              "bin_lon_center": ("bin", (edges[:-1] + edges[1:]) / 2)},
                             attrs={"n_bins": 20})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "geometry.nc"
            dataset.to_netcdf(path)
            data = regional_plot_data(self.summary, path)
            self.assertEqual(len(data), 40)
            self.assertEqual(float(data.mae_increase_pct_clipped.min()), 0.0)
            with self.assertRaises(ValueError):
                regional_plot_data(self.summary.loc[~self.summary.bin_index.eq(20)], path)


if __name__ == "__main__":
    unittest.main()
