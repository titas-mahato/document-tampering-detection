"""
Convolutional Denoising Autoencoder for Unsupervised Anomaly Detection
Directly aligns with Course Module 5 (Generative Models: Autoencoders & Dimensionality Reduction).
"""

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_autoencoder(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    latent_dim: int = 128
) -> tf.keras.Model:
    """
    Constructs an unsupervised Convolutional Autoencoder.

    Trained exclusively on authentic document samples to learn the nominal distribution
    of legitimate document text, background texture, and typography. At test time,
    forged or spliced regions produce high reconstruction residuals.

    Args:
        input_shape: Input image dimensions (H, W, C).
        latent_dim: Bottleneck representation channel depth.

    Returns:
        Full Autoencoder tf.keras.Model instance.
    """
    inputs = layers.Input(shape=input_shape, name="ae_input")

    # ==========================
    # ENCODER (Compression to Latent Space)
    # ==========================
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 128x128

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 64x64

    x = layers.Conv2D(latent_dim, (3, 3), activation="relu", padding="same", name="latent_bottleneck")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 32x32

    # ==========================
    # DECODER (Reconstruction)
    # ==========================
    x = layers.Conv2D(latent_dim, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # 64x64

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # 128x128

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # 256x256

    decoded = layers.Conv2D(input_shape[2], (3, 3), activation="sigmoid", padding="same", name="reconstruction")(x)

    model = models.Model(inputs=inputs, outputs=decoded, name="document_anomaly_autoencoder")
    return model


def compute_reconstruction_error(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Computes pixel-wise squared reconstruction error between input and autoencoder output.

    Args:
        model: Trained autoencoder.
        image_tensor: Normalized float32 numpy image array (H, W, 3) in range [0, 1].
        target_size: Autoencoder expected input dimensions.

    Returns:
        2D heatmap array (H, W) where high values indicate anomalous / tampered areas.
    """
    if image_tensor.ndim == 3:
        batch_input = np.expand_dims(image_tensor, axis=0)
    else:
        batch_input = image_tensor

    reconstructed = model.predict(batch_input, verbose=0)
    diff = np.square(batch_input[0] - reconstructed[0])
    error_map = np.mean(diff, axis=-1)
    return error_map
