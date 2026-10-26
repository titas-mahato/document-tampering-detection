"""
Unit tests for Neural Network architectures and loss/metrics.
Supports both unittest (standard library) and pytest.
"""

import unittest
import numpy as np

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


class TestModels(unittest.TestCase):

    @unittest.skipUnless(HAS_TF, "TensorFlow not installed in current environment")
    def test_unet_output_shape(self):
        from src.models.unet import build_unet
        model = build_unet(input_shape=(64, 64, 3), start_filters=16)
        dummy_input = np.zeros((1, 64, 64, 3), dtype=np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 64, 64, 1))

    @unittest.skipUnless(HAS_TF, "TensorFlow not installed in current environment")
    def test_autoencoder_output_shape(self):
        from src.models.autoencoder import build_autoencoder
        model = build_autoencoder(input_shape=(64, 64, 3), latent_dim=32)
        dummy_input = np.zeros((1, 64, 64, 3), dtype=np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 64, 64, 3))

    @unittest.skipUnless(HAS_TF, "TensorFlow not installed in current environment")
    def test_dice_coef_identical(self):
        from src.utils.metrics import dice_coef
        y_true = tf.constant([[[[1.0], [1.0]], [[0.0], [0.0]]]])
        y_pred = tf.constant([[[[1.0], [1.0]], [[0.0], [0.0]]]])
        dice = dice_coef(y_true, y_pred).numpy()
        self.assertGreater(float(dice), 0.99)

    @unittest.skipUnless(HAS_TF, "TensorFlow not installed in current environment")
    def test_iou_metric_identical(self):
        from src.utils.metrics import iou_metric
        y_true = tf.constant([[[[1.0], [1.0]], [[0.0], [0.0]]]])
        y_pred = tf.constant([[[[1.0], [1.0]], [[0.0], [0.0]]]])
        iou = iou_metric(y_true, y_pred).numpy()
        self.assertGreater(float(iou), 0.99)


if __name__ == "__main__":
    unittest.main()
