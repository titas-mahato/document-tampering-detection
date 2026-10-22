"""
U-Net Segmentation Network for Pixel-Level Tampering Localization
Implements an Encoder-Decoder architecture with skip connections.
Directly aligns with Course Module 3 (Spatial Vision & Segmentation).
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def _conv_block(inputs, filters: int, dropout_rate: float = 0.0):
    """Double convolution block with Batch Normalization and ReLU."""
    x = layers.Conv2D(filters, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    start_filters: int = 32,
    num_classes: int = 1
) -> tf.keras.Model:
    """
    Constructs a U-Net architecture for localizing tampered document regions.

    Skip connections preserve fine-grained spatial micro-edges from the encoder
    to the expanding decoder, enabling accurate segmentation of small modified text regions.

    Args:
        input_shape: (H, W, C) input dimensions (default: 512x512x3).
        start_filters: Base number of filters in the initial layer.
        num_classes: Output channels (1 for binary tamper mask with sigmoid).

    Returns:
        U-Net tf.keras.Model instance.
    """
    inputs = layers.Input(shape=input_shape, name="unet_input")

    # ==========================
    # ENCODER (Contracting Path)
    # ==========================
    c1 = _conv_block(inputs, start_filters)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = _conv_block(p1, start_filters * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = _conv_block(p2, start_filters * 4)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = _conv_block(p3, start_filters * 8)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # ==========================
    # BOTTLENECK
    # ==========================
    bn = _conv_block(p4, start_filters * 16, dropout_rate=0.3)

    # ==========================
    # DECODER (Expanding Path with Skip Connections)
    # ==========================
    u4 = layers.Conv2DTranspose(start_filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])
    c5 = _conv_block(u4, start_filters * 8)

    u3 = layers.Conv2DTranspose(start_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = _conv_block(u3, start_filters * 4)

    u2 = layers.Conv2DTranspose(start_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = _conv_block(u2, start_filters * 2)

    u1 = layers.Conv2DTranspose(start_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = _conv_block(u1, start_filters)

    # Final 1x1 Convolution mapping to binary mask
    outputs = layers.Conv2D(num_classes, (1, 1), activation="sigmoid", name="tamper_mask")(c8)

    model = models.Model(inputs=inputs, outputs=outputs, name="unet_document_tamper_segmentor")
    return model
