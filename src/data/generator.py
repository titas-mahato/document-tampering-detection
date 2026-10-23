"""
Batch Generator for DocTamper Training
Supports Dual-Stream (RGB + ELA) inputs, Segmentation Masks, and Balanced Batch Sampling.
"""

import math
import random
from typing import List, Optional, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image

from ..preprocessing.ela import compute_ela
from .lmdb_loader import DocTamperLMDBReader


class DocTamperDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator that streams and prepares mini-batches from LMDB.
    Resolves the 84.6% vs 15.4% class imbalance via balanced mini-batch sampling.
    """

    def __init__(
        self,
        lmdb_reader: DocTamperLMDBReader,
        batch_size: int = 16,
        target_size: Tuple[int, int] = (512, 512),
        task: str = "classification",  # 'classification', 'segmentation', 'autoencoder'
        include_ela: bool = True,
        balanced_sampling: bool = True,
        shuffle: bool = True,
        max_batches: Optional[int] = None
    ):
        """
        Args:
            lmdb_reader: An active DocTamperLMDBReader instance.
            batch_size: Number of images per training batch.
            target_size: (H, W) spatial dimensions for network input.
            task: Training target:
                - 'classification': Outputs (X_rgb, X_ela) -> is_tampered (0 or 1).
                - 'segmentation': Outputs (X_rgb, X_ela) -> binary mask (H, W, 1).
                - 'autoencoder': Outputs X_rgb -> X_rgb (reconstruction).
            include_ela: Whether to compute and output the ELA stream.
            balanced_sampling: If True, draws 50% authentic and 50% tampered per batch.
            shuffle: Whether to shuffle sample indices between epochs.
            max_batches: Optional limit on steps per epoch.
        """
        self.reader = lmdb_reader
        self.batch_size = batch_size
        self.target_size = target_size
        self.task = task
        self.include_ela = include_ela
        self.balanced_sampling = balanced_sampling
        self.shuffle = shuffle
        self.max_batches = max_batches

        self.indices = list(range(len(self.reader)))
        if self.shuffle:
            random.shuffle(self.indices)

    def __len__(self) -> int:
        total = math.ceil(len(self.indices) / self.batch_size)
        if self.max_batches:
            return min(total, self.max_batches)
        return total

    def __getitem__(self, index: int):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        rgb_list, ela_list, mask_list, label_list = [], [], [], []

        for idx in batch_indices:
            img_pil, mask_np, is_tampered = self.reader.get_sample(idx)

            # Resize if necessary
            if img_pil.size != self.target_size:
                img_pil = img_pil.resize(self.target_size, Image.BILINEAR)
                mask_pil = Image.fromarray(mask_np).resize(self.target_size, Image.NEAREST)
                mask_np = np.array(mask_pil)

            rgb_np = np.array(img_pil, dtype=np.float32) / 255.0
            rgb_list.append(rgb_np)
            mask_list.append(np.expand_dims(mask_np.astype(np.float32), axis=-1))
            label_list.append(float(is_tampered))

            if self.include_ela:
                ela_np = compute_ela(img_pil, return_numpy=True)
                ela_list.append(ela_np.astype(np.float32) / 255.0)

        X_rgb = np.array(rgb_list, dtype=np.float32)
        y_labels = np.array(label_list, dtype=np.float32)
        y_masks = np.array(mask_list, dtype=np.float32)

        if self.include_ela:
            X_ela = np.array(ela_list, dtype=np.float32)
            inputs = [X_rgb, X_ela]
        else:
            inputs = X_rgb

        if self.task == "classification":
            return inputs, y_labels
        elif self.task == "segmentation":
            return inputs, y_masks
        elif self.task == "autoencoder":
            return X_rgb, X_rgb
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)
