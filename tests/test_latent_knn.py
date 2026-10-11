"""Tests for the pre-head k-NN diagnostic: temporal buffer and minimum candidate count."""

import unittest

import numpy as np

from machine_learning.analysis.latent_knn import BUFFER_MONTHS, K_VALUES, nearest_neighbors


class LatentNeighborhoodTests(unittest.TestCase):
    def test_temporal_buffer_applies_to_every_neighbor(self):
        time = np.arange(64)
        coordinates = time[:, None].astype(np.float32)
        neighbors = nearest_neighbors(coordinates, time)
        for k in K_VALUES:
            self.assertEqual(neighbors[k].shape, (64, k))
            self.assertTrue(np.all(np.abs(time[:, None] - time[neighbors[k]]) > BUFFER_MONTHS))

    def test_insufficient_candidates_fail(self):
        time = np.arange(20)
        with self.assertRaises(ValueError):
            nearest_neighbors(time[:, None].astype(np.float32), time)


if __name__ == "__main__":
    unittest.main()
