"""
Unit tests for Neural Network architectures and loss/metrics.
Supports both unittest (standard library) and pytest.
"""

import sys
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestModels(unittest.TestCase):

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in current environment")
    def test_dual_stream_cnn_forward(self):
        from src.models.cnn_classifier import DualStreamCNN
        model = DualStreamCNN(num_classes=2)
        model.eval()

        dummy_rgb = torch.zeros(2, 3, 224, 224)
        dummy_ela = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy_rgb, dummy_ela)

        self.assertEqual(out.shape, (2, 2))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed in current environment")
    def test_resnet34_unet_forward(self):
        from src.models.unet import ResNet34UNet
        model = ResNet34UNet(pretrained=False)
        model.eval()

        dummy_img = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            out = model(dummy_img)

        self.assertEqual(out.shape, (1, 1, 256, 256))


if __name__ == "__main__":
    unittest.main()
