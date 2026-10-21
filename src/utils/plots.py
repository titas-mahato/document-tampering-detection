"""
Visualization utilities for forensic diagnostics, side-by-side comparisons, and training curves.
"""

from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_forensic_analysis(
    original_img: np.ndarray,
    ela_img: np.ndarray,
    edges_img: np.ndarray,
    mask_pred: Optional[np.ndarray] = None,
    gradcam_overlay: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Plots a multi-panel forensic breakdown comparing original document,
    compression residuals, edge filtering, and neural network predictions.
    """
    panels = [
        ("Original Document", original_img, None),
        ("ELA Compression Map", ela_img, None),
        ("Laplacian Micro-Edges", edges_img, "gray"),
    ]

    if mask_pred is not None:
        panels.append(("Predicted Tamper Mask", mask_pred, "gray"))

    if gradcam_overlay is not None:
        panels.append(("Grad-CAM Attention", gradcam_overlay, None))

    num_panels = len(panels)
    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4.5))
    if num_panels == 1:
        axes = [axes]

    for ax, (title, img, cmap) in zip(axes, panels):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_training_history(history, metrics=("loss", "dice_coef"), save_path: Optional[str] = None):
    """
    Plots training vs validation curves for model diagnostics (Module 2: Bias-Variance tradeoff).
    """
    hist = history.history if hasattr(history, "history") else history
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 4.5))
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in hist:
            ax.plot(hist[metric], label=f"Train {metric.title()}", linewidth=2)
        val_metric = f"val_{metric}"
        if val_metric in hist:
            ax.plot(hist[val_metric], label=f"Val {metric.title()}", linestyle="--", linewidth=2)

        ax.set_title(f"Model {metric.replace('_', ' ').title()}", fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
