"""
Custom Losses and Evaluation Metrics for Document Tampering Detection
Directly aligns with Course Module 1 (Metrics) and Module 2 (Loss Functions & Imbalance).
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """
    Computes the Dice Similarity Coefficient (F1-score for pixel segmentation).
    Formula: 2 * |X ∩ Y| / (|X| + |Y|)
    """
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice Loss for training segmentation models on sparse masks (<10% altered pixels)."""
    return 1.0 - dice_coef(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth: float = 1e-6):
    """
    Computes Intersection over Union (IoU / Jaccard Index) for binary masks.
    Formula: |X ∩ Y| / |X ∪ Y|
    """
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred > 0.5, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def bce_dice_loss(y_true, y_pred, bce_weight: float = 0.5):
    """
    Hybrid loss combining Binary Cross-Entropy with Soft Dice Loss.
    Addresses extreme spatial pixel class imbalance.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss for binary classification or dense mask prediction.
    Down-weights easy well-classified background pixels to focus gradient updates on hard tampering edges.
    """
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # p_t: probability of true class
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating_factor = K.pow(1.0 - p_t, gamma)
        
        loss = -alpha_factor * modulating_factor * K.log(p_t)
        return K.mean(loss)
    
    return focal_loss_fn
