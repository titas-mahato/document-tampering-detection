"""
Neural network architectures for document tampering classification and localization.
Implements Dual-Stream CNN (CASIA) and ResNet34 U-Net (DocTamper).
"""

from .cnn_classifier import DualStreamCNN, CNNFeatureExtractor, build_dual_stream_classifier
from .unet import ResNet34UNet, DecoderBlock, DoubleConv, build_unet
from .autoencoder import build_autoencoder, compute_reconstruction_error

__all__ = [
    "DualStreamCNN",
    "CNNFeatureExtractor",
    "build_dual_stream_classifier",
    "ResNet34UNet",
    "DecoderBlock",
    "DoubleConv",
    "build_unet",
    "build_autoencoder",
    "compute_reconstruction_error"
]
