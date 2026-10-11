"""Focused checks for the train-only seasonal-climatology baseline."""

import unittest
from types import SimpleNamespace

import numpy as np

from machine_learning.analysis.baselines import climatology_predictions


class BaselineRulesTest(unittest.TestCase):
    def test_climatology_uses_training_months_only(self):
        train = SimpleNamespace(time_index=np.array(["2000-01-16", "2001-01-16", "2000-02-15"],
                                                  dtype="datetime64[D]"),
                                y_phys=np.array([2.0, 4.0, 8.0]))
        test = SimpleNamespace(time_index=np.array(["2002-01-16", "2002-02-15", "2002-03-16"],
                                                 dtype="datetime64[D]"))
        np.testing.assert_array_equal(climatology_predictions(train, test, "M"), [3.0, 8.0, 14 / 3])

    def test_quarterly_climatology_uses_calendar_quarters(self):
        train = SimpleNamespace(time_index=np.array(["2000-03-16", "2000-06-15", "2001-03-16"],
                                                  dtype="datetime64[D]"),
                                y_phys=np.array([2.0, 9.0, 4.0]))
        test = SimpleNamespace(time_index=np.array(["2002-03-16", "2002-06-15"],
                                                 dtype="datetime64[D]"))
        np.testing.assert_array_equal(climatology_predictions(train, test, "Q"), [3.0, 9.0])


if __name__ == "__main__":
    unittest.main()
