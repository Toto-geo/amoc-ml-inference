"""Focused checks for the loss function, checkpoint-tie decision and RunConfig validation."""

import unittest

import torch

from machine_learning.train import RunConfig, is_better, loss_fn


class TrainingRulesTest(unittest.TestCase):
    def test_paper_loss(self):
        prediction = torch.tensor([1.0, 2.0, 4.0])
        target = torch.tensor([0.0, 3.0, 5.0])
        eps = 1e-6
        expected = torch.mean((prediction - target) ** 2) + 0.1 * torch.log(
            torch.sqrt(torch.mean((prediction - prediction.mean()) ** 2) + eps)
            / torch.sqrt(torch.mean((target - target.mean()) ** 2) + eps)
        ) ** 2
        torch.testing.assert_close(loss_fn(prediction, target), expected)

    def test_validation_tie_break(self):
        self.assertTrue(is_better(0.99, 0.8, 1.0, 0.6))
        self.assertTrue(is_better(0.998, 0.5, 1.0, 0.6))
        self.assertFalse(is_better(0.998, 0.7, 1.0, 0.6))
        self.assertFalse(is_better(1.01, 0.5, 1.0, 0.6))

    def test_only_paper_seeds_and_resolutions(self):
        RunConfig("Q", 8, 65537, 160).validate()
        self.assertEqual(RunConfig("M", 1, 42).center_init, "kmeans")
        RunConfig("M", 1, 42, center_init="kmeans").validate()
        with self.assertRaises(ValueError):
            RunConfig("Q", 8, 8, 160).validate()
        with self.assertRaises(ValueError):
            RunConfig("Q", 8, 42, 30).validate()
        with self.assertRaises(ValueError):
            RunConfig("M", 1, 42, center_init="random").validate()


if __name__ == "__main__":
    unittest.main()
