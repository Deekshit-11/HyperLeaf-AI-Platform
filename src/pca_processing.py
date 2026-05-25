"""
pca_processing.py — HyperLeaf Pro
PCA dimensionality reduction: 204 bands → 50 components.
Fits on training data, persists transformer, applies at inference.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PCA_PATH    = Path(__file__).parent.parent / "data" / "pca_model.pkl"
SCALER_PATH = Path(__file__).parent.parent / "data" / "scaler.pkl"
N_COMPONENTS = 50


def fit_pca(X: np.ndarray, n_components: int = N_COMPONENTS):
    """
    Fit PCA + StandardScaler on training data.

    Args:
        X: (n_samples, n_bands)
        n_components: number of PCA components

    Returns:
        X_pca: (n_samples, n_components)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Persist
    PCA_PATH.parent.mkdir(exist_ok=True)
    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[PCA] Fitted: {X.shape[1]} → {n_components} components")
    print(f"[PCA] Variance retained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    return X_pca


def apply_pca(x: np.ndarray) -> np.ndarray:
    """
    Apply fitted PCA to a single sample or batch.

    Args:
        x: (n_bands,) or (n_samples, n_bands)

    Returns:
        x_pca: (n_components,) or (n_samples, n_components)
    """
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    x_scaled = scaler.transform(x)
    x_pca    = pca.transform(x_scaled)

    return x_pca[0] if single else x_pca


def get_explained_variance() -> np.ndarray:
    """Return explained variance ratio array for visualization."""
    if not PCA_PATH.exists():
        return np.array([])
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    return pca.explained_variance_ratio_


def pca_available() -> bool:
    return PCA_PATH.exists() and SCALER_PATH.exists()