"""
Unit tests for ELA and Laplacian edge preprocessing modules.
Supports both unittest (standard library) and pytest.
"""

import unittest
import numpy as np
from PIL import Image

from src.preprocessing.ela import compute_ela, compute_ela_mean_energy
from src.preprocessing.edges import compute_laplacian_edges


class TestPreprocessing(unittest.TestCase):

    def test_compute_ela_pil_input(self):
        img = Image.new("RGB", (128, 128), color=(200, 150, 100))
        ela = compute_ela(img, quality=90, multiplier=25.0, return_numpy=False)

        self.assertIsInstance(ela, Image.Image)
        self.assertEqual(ela.size, (128, 128))

    def test_compute_ela_numpy_output(self):
        img_np = np.full((64, 64, 3), 180, dtype=np.uint8)
        ela_np = compute_ela(img_np, quality=90, multiplier=25.0, return_numpy=True)

        self.assertIsInstance(ela_np, np.ndarray)
        self.assertEqual(ela_np.shape, (64, 64, 3))
        self.assertEqual(ela_np.dtype, np.uint8)

    def test_compute_ela_mean_energy(self):
        ela_np = np.full((50, 50, 3), 15, dtype=np.uint8)
        energy = compute_ela_mean_energy(ela_np)
        self.assertIsInstance(energy, float)
        self.assertAlmostEqual(energy, 15.0, places=3)

    def test_compute_laplacian_edges(self):
        img_np = np.zeros((100, 100, 3), dtype=np.uint8)
        img_np[30:70, 30:70] = 255

        edges = compute_laplacian_edges(img_np, normalize=True)

        self.assertIsInstance(edges, np.ndarray)
        self.assertEqual(edges.shape, (100, 100))
        self.assertEqual(edges.dtype, np.uint8)
        self.assertGreater(int(np.max(edges)), 0)


if __name__ == "__main__":
    unittest.main()
