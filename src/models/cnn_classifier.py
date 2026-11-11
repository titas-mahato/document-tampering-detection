"""
Dual-Stream CNN Classifier for Document Tampering Detection
Combines RGB visual features with ELA compression artifact features.
Corresponds to Course Module 3 (CNNs & Multi-modal Architecture) and Module 2 (Regularization).
"""

from typing import Optional
import os

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class CNNFeatureExtractor(nn.Module):
        """
        4-block Convolutional feature extractor with Batch Normalization,
        Max Pooling, and Adaptive Average Pooling.
        """
        def __init__(self, in_channels: int = 3):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 4
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.AdaptiveAvgPool2d((1, 1))
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            return torch.flatten(x, 1)


    class DualStreamCNN(nn.Module):
        """
        Dual-Stream Convolutional Neural Network for Forgery Classification.
        - Stream A: RGB visual features (layout, typography, characters)
        - Stream B: ELA compression error residuals (re-compression discrepancies)
        - Fusion: Concatenates both feature vectors (256 + 256 = 512) into a Dense Classifier.
        """
        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.rgb_branch = CNNFeatureExtractor(in_channels=3)
            self.ela_branch = CNNFeatureExtractor(in_channels=3)

            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),

                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),

                nn.Linear(128, num_classes)
            )

        def forward(self, rgb: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
            rgb_f = self.rgb_branch(rgb)
            ela_f = self.ela_branch(ela)
            fused = torch.cat((rgb_f, ela_f), dim=1)
            return self.classifier(fused)


    def build_dual_stream_classifier(num_classes: int = 2, weights_path: Optional[str] = None, device: str = "cpu") -> DualStreamCNN:
        """Instantiates and optionally loads pretrained weights into DualStreamCNN."""
        model = DualStreamCNN(num_classes=num_classes)
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        model.to(device)
        return model

else:
    class CNNFeatureExtractor:
        pass

    class DualStreamCNN:
        pass

    def build_dual_stream_classifier(*args, **kwargs):
        raise ImportError("PyTorch is required to build DualStreamCNN. Install with `pip install torch`.")
