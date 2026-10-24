"""dHash helpers shared by check_near_duplicates.py and build_ppe_dataset.py."""
import numpy as np
from PIL import Image


def dhash(path):
    """64-bit difference hash as 8 packed bytes."""
    a = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def hamming(a, B):
    """Hamming distance between one hash and a stack of hashes."""
    return np.unpackbits(a[None, :] ^ B, axis=1).sum(axis=1)
