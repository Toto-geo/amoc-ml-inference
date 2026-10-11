"""Checks for the post-hoc, full-record frequency decomposition and target structure."""

import unittest

import numpy as np
import pandas as pd

from machine_learning.analysis.temporal import decompose, target_structure


class TemporalRulesTest(unittest.TestCase):
    def test_monthly_uses_centered_thirteen_month_window(self):
        dates = pd.date_range("2000-01-01", periods=26, freq="MS")
        values = np.arange(26, dtype=float) + np.sin(np.arange(26))
        part = decompose(values, dates, "M")
        expected = part.deseasonalized_anomaly.rolling(13, center=True, min_periods=1).mean()
        np.testing.assert_allclose(part.low_pass, expected)
        np.testing.assert_allclose(part.seasonal_cycle + part.low_pass + part.high_frequency,
                                   values, atol=1e-14)

    def test_quarterly_uses_five_quarter_window(self):
        dates = pd.date_range("2000-03-01", periods=12, freq="QS-MAR")
        values = np.arange(12, dtype=float)
        part = decompose(values, dates, "Q")
        np.testing.assert_allclose(part.low_pass,
                                   part.deseasonalized_anomaly.rolling(5, center=True,
                                                                        min_periods=1).mean())

    def test_unordered_dates_are_rejected(self):
        dates = pd.DatetimeIndex(["2000-02-01", "2000-01-01"])
        with self.assertRaisesRegex(ValueError, "unique and ordered"):
            decompose(np.array([1.0, 2.0]), dates, "M")

    def test_monthly_structure_uses_complete_centered_windows(self):
        months = np.arange(816)
        frame = pd.DataFrame({"regime": "M", "bins": 20,
                              "global_time_index": months,
                              "true_high_frequency": np.sin(2 * np.pi * months / 8)})
        acf, ljung_box, spectrum, bands = target_structure(frame)
        self.assertEqual(set(acf.n_samples), {804})
        self.assertEqual(len(acf), 12)
        self.assertEqual(set(ljung_box.lag_months), {6, 12})
        self.assertEqual(len(spectrum), 65)
        self.assertEqual(len(bands), 4)


if __name__ == "__main__":
    unittest.main()
