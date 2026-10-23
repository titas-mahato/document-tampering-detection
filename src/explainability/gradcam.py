"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module
Provides visual interpretability by highlighting image regions that drive the tamper classification decision.
Directly implements Selvaraju et al. (ICCV 2017) as cited in Course Module 3 syllabus.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import tensorflow as tf


def compute_gradcam(
    model: tf.keras.Model,
    input_tensors: Union[np.ndarray, list],
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Computes a Grad-CAM heatmap for a target convolutional layer.

    Args:
        model: Trained Keras classification model.
        input_tensors: Model input(s) with batch dimension (e.g. [1, 512, 512, 3]).
        last_conv_layer_name: Name of the convolutional feature layer. If None,
                              automatically locates the last Conv2D layer.
        pred_index: Target output class index (None for binary sigmoid).

    Returns:
        2D heatmap array normalized to [0, 1].
    """
    # Auto-detect last Conv2D layer if name is not explicitly provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            # Check for Conv2D layer directly or nested within a sub-model (e.g. MobileNetV2)
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
            elif isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv_layer_name = sub_layer.name
                        break
                if last_conv_layer_name:
                    break

    if not last_conv_layer_name:
        raise ValueError("Could not find a convolutional layer in the provided model.")

    # Build sub-model that maps model inputs to (last conv activations, model predictions)
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )
    except ValueError:
        # Layer might be inside a nested functional backbone
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    conv_layer = layer.get_layer(last_conv_layer_name)
                    grad_model = tf.keras.models.Model(
                        inputs=model.inputs,
                        outputs=[conv_layer.output, model.output]
                    )
                    break
                except ValueError:
                    continue

    # Record operations to compute gradients via tf.GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensors)
        if pred_index is None:
            target_class_channel = predictions[:, 0]
        else:
            target_class_channel = predictions[:, pred_index]

    # Compute gradient of target class score with respect to feature maps
    grads = tape.gradient(target_class_channel, conv_outputs)

    # Global average pooling over gradients to get feature importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only positive contributions toward target class are kept
    heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto the original RGB document image.

    Args:
        original_image: RGB uint8 numpy array (H, W, 3).
        heatmap: 2D float numpy array in range [0, 1].
        alpha: Blending weight for the colormap overlay.
        colormap: OpenCV colormap constant (default: COLORMAP_JET).

    Returns:
        Blended RGB uint8 image.
    """
    h, w = original_image.shape[:2]
    # Resize heatmap to match original document dimensions
    resized_heatmap = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * resized_heatmap)

    # Apply colormap
    color_map = cv2.applyColorMap(heatmap_uint8, colormap)
    color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # Weighted blend
    blended = np.clip(
        alpha * color_map_rgb.astype(np.float32) + (1.0 - alpha) * original_image.astype(np.float32),
        0,
        255
    ).astype(np.uint8)

    return blended
