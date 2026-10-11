"""Checks for the synthetic minimal end-to-end data contract."""

from pathlib import Path
import tempfile
import unittest

from examples.minimal_end_to_end.generate_data import generate
from machine_learning.dataset import build_datasets


class MinimalEndToEndTest(unittest.TestCase):
    def test_generated_data_loads_for_both_regimes(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = generate(Path(directory) / "data")
            expected = {"M": (612, 102, 102), "Q": (204, 34, 34)}
            for regime, lengths in expected.items():
                train, val, test, stats = build_datasets(
                    regime, 1, n_bins=20, data_root=data_root
                )
                self.assertEqual((len(train), len(val), len(test)), lengths)
                self.assertEqual(stats["P"], 32)
                self.assertEqual(stats["F"], 54)
                self.assertEqual(stats["surface_n_bins"], 20)


if __name__ == "__main__":
    unittest.main()
