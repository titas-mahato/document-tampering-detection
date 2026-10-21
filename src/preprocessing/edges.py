"""
Laplacian Edge Filtering Module
Extracts high-frequency micro-edge noise to reveal digital splicing boundaries.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image


def compute_laplacian_edges(
    image: Union[Image.Image, np.ndarray, str, Path],
    normalize: bool = True
) -> np.ndarray:
    """
    Applies an 8-neighbor high-pass Laplacian convolution kernel to extract
    micro-edge artifacts around spliced text, dates, or inserted image regions.

    Args:
        image: PIL Image, RGB/grayscale numpy array, or file path.
        normalize: Whether to normalize output to full 0-255 uint8 range.

    Returns:
        Grayscale edge map as uint8 numpy array (H, W).
    """
    if isinstance(image, (str, Path)):
        img_np = cv2.imread(str(image))
        if img_np is None:
            raise FileNotFoundError(f"Failed to load image from: {image}")
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:
            gray = image.astype(np.uint8)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # 8-neighbor discrete Laplacian high-pass kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32) / 8.0

    edges = cv2.filter2D(gray, -1, kernel)

    if normalize:
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        edges = np.clip(edges, 0, 255).astype(np.uint8)

    return edges
