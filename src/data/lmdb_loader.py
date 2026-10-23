"""
DocTamper LMDB Dataset Loader
Directly streams paired document images and binary tampering masks from LMDB databases.
"""

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import lmdb
import numpy as np
from PIL import Image


class DocTamperLMDBReader:
    """
    Direct reader for DocTamper LMDB database instances.
    Provides memory-efficient key indexing and on-the-fly sample decoding.
    """

    def __init__(self, lmdb_path: str, max_samples: Optional[int] = None):
        """
        Args:
            lmdb_path: Directory path containing the DocTamper data.mdb file.
            max_samples: Optional cap on indexed keys (useful for quick local testing/EDA).
        """
        self.lmdb_path = str(lmdb_path)
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB directory not found: {self.lmdb_path}")

        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=32
        )

        self.image_keys: List[bytes] = []
        self._index_keys(max_samples)

    def _index_keys(self, max_samples: Optional[int] = None):
        """Indexes all primary image keys starting with b'image-'."""
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(keys=True, values=False):
                if key.startswith(b"image-"):
                    self.image_keys.append(key)
                    if max_samples and len(self.image_keys) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.image_keys)

    def get_sample(self, idx: int) -> Tuple[Image.Image, np.ndarray, int]:
        """
        Retrieves a document sample by integer index.

        Returns:
            Tuple of:
            - image_pil: Document image as RGB PIL Image.
            - mask_np: Binary ground truth mask (H, W) uint8 (1 = tampered, 0 = authentic).
            - is_tampered: Binary class label (1 if any pixel tampered, else 0).
        """
        key = self.image_keys[idx]
        lbl_key = key.replace(b"image-", b"label-")

        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(key)
            lbl_bytes = txn.get(lbl_key)

        if img_bytes is None:
            raise KeyError(f"Image key not found in LMDB: {key}")
        if lbl_bytes is None:
            raise KeyError(f"Label mask key not found in LMDB: {lbl_key}")

        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mask_raw = Image.open(io.BytesIO(lbl_bytes)).convert("L")
        mask_np = (np.array(mask_raw) > 127).astype(np.uint8)

        is_tampered = int(np.sum(mask_np) > 0)
        return image_pil, mask_np, is_tampered

    def close(self):
        """Closes the LMDB environment safely."""
        try:
            self.env.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
