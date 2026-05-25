"""
preprocessing.py — HyperLeaf Pro
Hyperspectral image preprocessing with augmentation pipeline.
Handles .tif/.tiff/.npy inputs.
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional

# ─── Water absorption bands to remove (indices for 204-band 400-2500nm range) ───
WATER_ABSORPTION_BANDS = list(range(100, 115)) + list(range(148, 163))  # ~1400nm, ~1900nm

def load_hyperspectral(filepath: str) -> Optional[np.ndarray]:
    """Load a hyperspectral image from .tif, .tiff, or .npy."""
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext in [".npy"]:
        data = np.load(filepath)
        return data

    if ext in [".tif", ".tiff"]:
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                data = src.read()          # shape: (bands, H, W)
                data = data.astype(np.float32)
                return data
        except ImportError:
            try:
                import spectral
                img = spectral.open_image(filepath)
                data = img.load()          # shape: (H, W, bands)
                data = np.transpose(data, (2, 0, 1))
                return data.astype(np.float32)
            except Exception as e:
                raise RuntimeError(f"Could not load TIFF: {e}")

    raise ValueError(f"Unsupported file format: {ext}")


def remove_water_bands(data: np.ndarray) -> np.ndarray:
    """Remove water absorption bands from spectrum."""
    n_bands = data.shape[0] if data.ndim == 3 else data.shape[-1]
    all_bands = list(range(n_bands))
    valid_bands = [b for b in all_bands if b not in WATER_ABSORPTION_BANDS]
    if data.ndim == 3:
        return data[valid_bands, :, :]
    else:
        return data[..., valid_bands]


def normalize_bands(data: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization."""
    if data.ndim == 3:
        bands, H, W = data.shape
        out = np.zeros_like(data, dtype=np.float32)
        for b in range(bands):
            band = data[b]
            mn, mx = band.min(), band.max()
            out[b] = (band - mn) / (mx - mn + 1e-8)
        return out
    else:
        mn = data.min(axis=-1, keepdims=True)
        mx = data.max(axis=-1, keepdims=True)
        return (data - mn) / (mx - mn + 1e-8)


def spatial_mean_pooling(data: np.ndarray) -> np.ndarray:
    """Reduce (bands, H, W) → (bands,) by spatial mean."""
    if data.ndim == 3:
        return data.mean(axis=(1, 2))
    elif data.ndim == 2:
        return data.mean(axis=0)
    return data


def preprocess(filepath_or_array, return_spatial: bool = False):
    """
    Full preprocessing pipeline.

    Args:
        filepath_or_array: path string or numpy array
        return_spatial: if True, return (bands, H, W) spatial cube for viz

    Returns:
        1D feature vector of shape (n_valid_bands,)  — ready for PCA
        OR spatial cube (bands, H, W) if return_spatial=True
    """
    if isinstance(filepath_or_array, (str, Path)):
        data = load_hyperspectral(str(filepath_or_array))
    else:
        data = filepath_or_array.astype(np.float32)
        if data.ndim == 1:
            # Already a flat feature vector
            return data

    # Ensure (bands, H, W)
    if data.ndim == 2:
        data = data[..., np.newaxis]         # (bands, 1)
    if data.shape[0] < data.shape[-1]:       # (H, W, bands) → (bands, H, W)
        data = np.transpose(data, (2, 0, 1))

    data = remove_water_bands(data)
    data = normalize_bands(data)

    if return_spatial:
        return data                          # (bands, H, W)

    return spatial_mean_pooling(data)        # (bands,)


# ─── Augmentation ──────────────────────────────────────────────────────────────

def augment_spectral(feature_vector: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Spectral augmentation for 1D feature vectors.
    Simulates: Gaussian noise, random scaling, band dropout.
    """
    rng = np.random.RandomState(seed)
    x = feature_vector.copy()

    # Gaussian noise
    x += rng.normal(0, 0.01, x.shape)

    # Random scaling
    x *= rng.uniform(0.95, 1.05)

    # Random band dropout (set 5% of bands to 0)
    dropout_mask = rng.rand(len(x)) < 0.05
    x[dropout_mask] = 0.0

    return np.clip(x, 0, 1).astype(np.float32)


def augment_spatial(cube: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Spatial augmentation for (bands, H, W) hyperspectral cubes.
    Supports: flip, crop, Gaussian noise.
    """
    rng = np.random.RandomState(seed)
    x = cube.copy()

    # Horizontal flip
    if rng.rand() > 0.5:
        x = x[:, :, ::-1]

    # Vertical flip
    if rng.rand() > 0.5:
        x = x[:, ::-1, :]

    # Gaussian noise
    x += rng.normal(0, 0.005, x.shape).astype(np.float32)

    return np.clip(x, 0, 1).astype(np.float32)