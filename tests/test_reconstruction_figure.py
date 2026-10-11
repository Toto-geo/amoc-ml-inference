"""Smoke tests for the eight-panel reconstruction figure generator (fig06)."""

import io
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.scripts.fig06_reconstruction import COLORS, build_figure


class ReconstructionFigureTests(unittest.TestCase):
    def test_paper_palette_and_eight_panels(self):
        self.assertEqual((COLORS["base"], COLORS["ridge"], COLORS["climatology"]),
                         ("#b22222", "#2f6b3f", "#7f7f7f"))
        series = []
        for regime, length, freq in (("M", 816, "MS"), ("Q", 272, "QS")):
            t = np.arange(length)
            series.append(pd.DataFrame({"regime": regime,
                                        "fold": t // (length // 8) + 1,
                                        "date": pd.date_range("1958-01-01", periods=length, freq=freq),
                                        "y_true_phys": 12 + np.sin(t / 6),
                                        "y_pred_phys": 12 + 0.6 * np.sin(t / 6)}))
        metrics = pd.DataFrame([{"series_id": name, "fold": fold,
                                 "rmse": 4.0, "mae": 3.0, "bias": 0.1,
                                 "pearson": 0.6, "r2": 0.2, "spread_ratio": 0.7}
                                for name in ("latent_M_full", "ridge_M", "climatology_M",
                                             "latent_Q_quarter_mean", "ridge_Q", "climatology_Q")
                                for fold in range(1, 9)])
        fig = build_figure(pd.concat(series, ignore_index=True), metrics)
        try:
            self.assertEqual(len(fig.axes), 9)  # 8 panels plus shared fold boundaries
            pdf = io.BytesIO()
            fig.savefig(pdf, format="pdf", bbox_inches="tight")
            self.assertTrue(pdf.getvalue().startswith(b"%PDF"))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
