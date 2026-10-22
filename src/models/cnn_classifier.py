"""
Dual-Stream CNN Classifier for Document Tampering Detection
Combines RGB visual features with ELA compression artifact features.
Corresponds to Course Module 3 (CNNs & Transfer Learning) and Module 2 (Regularization).
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_dual_stream_classifier(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    backbone_type: str = "mobilenetv2",
    pretrained: bool = True,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Constructs a dual-stream neural network for document tampering classification.

    - Stream A (RGB): Extracts deep spatial semantics (font geometry, structural alignment)
      using a transfer-learned MobileNetV2 backbone.
    - Stream B (ELA): Extracts high-frequency compression residuals through a dedicated
      convolutional stream.
    - Fusion: Concatenates feature vectors into dense regularized layers with Dropout.

    Args:
        input_shape: Dimensions of document inputs (H, W, C).
        backbone_type: Base architecture ('mobilenetv2' or 'resnet50').
        pretrained: Whether to load ImageNet weights for the RGB backbone.
        dropout_rate: Dropout probability for regularization (Module 2).
        l2_reg: L2 weight decay penalty factor.

    Returns:
        Compiled or uncompiled tf.keras.Model instance.
    """
    input_rgb = layers.Input(shape=input_shape, name="input_rgb")
    input_ela = layers.Input(shape=input_shape, name="input_ela")

    # ==========================
    # STREAM 1: RGB Visual Backbone
    # ==========================
    weights = "imagenet" if pretrained else None
    if backbone_type.lower() == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    elif backbone_type.lower() == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}")

    # Fine-tuning setup: freeze lower layers initially
    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    x_rgb = base_model(input_rgb)
    x_rgb = layers.GlobalAveragePooling2D(name="rgb_gap")(x_rgb)
    x_rgb = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x_rgb)

    # ==========================
    # STREAM 2: ELA Forensic Stream
    # ==========================
    x_ela = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu")(input_ela)
    x_ela = layers.BatchNormalization()(x_ela)
    x_ela = layers.Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x_ela)
    x_ela = layers.BatchNormalization()(x_ela)
    x_ela = layers.Conv2D(128, (3, 3), strides=2, padding="same", activation="relu")(x_ela)
    x_ela = layers.GlobalAveragePooling2D(name="ela_gap")(x_ela)
    x_ela = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x_ela)

    # ==========================
    # MULTI-MODAL FUSION & HEAD
    # ==========================
    fused = layers.Concatenate(name="feature_fusion")([x_rgb, x_ela])
    fused = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(dropout_rate, name="dropout_regularization")(fused)

    output = layers.Dense(1, activation="sigmoid", name="tamper_probability")(fused)

    model = models.Model(inputs=[input_rgb, input_ela], outputs=output, name="dual_stream_tamper_classifier")
    return model
