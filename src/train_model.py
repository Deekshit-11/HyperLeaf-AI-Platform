"""
train_model.py — HyperLeaf Pro
NitrogenCNN with:
  - Residual blocks for depth without vanishing gradients
  - Dropout + BatchNorm for regularization
  - K-Fold Cross Validation training
  - MixUp augmentation
  - Early stopping
  - Grad-CAM hook support
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import json, time

MODEL_DIR  = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "best_model.pth"
KFOLD_PATH = MODEL_DIR / "kfold_results.json"

# ─────────────────────────────────────────────────────────────────────────────
#  Architecture
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class NitrogenCNN(nn.Module):
    """
    1D CNN operating on PCA feature vectors (50-dim).
    Input shape: (batch, 1, 50)  [channel-first]
    Output: scalar sigmoid score in [0,1]
    """
    def __init__(self, input_dim: int = 50, dropout: float = 0.4):
        super().__init__()
        self.input_dim = input_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),       # 50 → 25
        )

        # Residual blocks
        self.layer1 = ResidualBlock1D(32, 64,  dropout=dropout)
        self.pool1  = nn.MaxPool1d(2)          # 25 → 12

        self.layer2 = ResidualBlock1D(64, 128, dropout=dropout)
        self.pool2  = nn.AdaptiveAvgPool1d(4)  # → 4

        # Classifier head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Grad-CAM hook targets
        self.gradients   = None
        self.activations = None
        self.layer2.conv2.register_forward_hook(self._save_activation)
        self.layer2.conv2.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)         # (B, 50) → (B, 1, 50)
        x = self.stem(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam(model: NitrogenCNN, x: torch.Tensor) -> np.ndarray:
    """
    Compute Grad-CAM weights for a single input.

    Args:
        model: NitrogenCNN (eval mode)
        x: (1, 50) or (1, 1, 50) tensor

    Returns:
        cam: normalized 1D array of shape (input_dim,) — importance per PCA dim
    """
    model.eval()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.requires_grad_(True)

    out = model(x)
    model.zero_grad()
    out.backward()

    grads = model.gradients           # (1, C, L)
    acts  = model.activations         # (1, C, L)

    weights = grads.mean(dim=-1, keepdim=True)    # global average pool
    cam = (weights * acts).sum(dim=1).squeeze()   # (L,)
    cam = F.relu(torch.tensor(cam)).numpy()

    # Upsample to input_dim
    cam = np.interp(
        np.linspace(0, 1, model.input_dim),
        np.linspace(0, 1, len(cam)),
        cam
    )
    # Normalize
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  MixUp
# ─────────────────────────────────────────────────────────────────────────────

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    return mixed_x, mixed_y


# ─────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = np.inf
        self.triggered = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
        return self.triggered


def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if use_mixup:
            xb, yb_mix = mixup_batch(xb, yb.float())
        else:
            yb_mix = yb.float()
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = criterion(pred, yb_mix)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        # accuracy on original labels
        correct += ((pred.detach() > 0.5).long() == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).squeeze()
        loss = criterion(pred, yb.float())
        total_loss += loss.item() * xb.size(0)
        preds_bin = (pred > 0.5).long()
        correct   += (preds_bin == yb).sum().item()
        total     += xb.size(0)
        all_preds.extend(pred.cpu().numpy().tolist())
        all_labels.extend(yb.cpu().numpy().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
#  K-Fold Training Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def train_kfold(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 3e-4,
    device_str: str = "auto",
):
    """
    K-Fold cross-validation training for NitrogenCNN.

    Returns:
        fold_results: list of dicts with accuracy, val_loss per fold
        best_model_path: path to saved best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    print(f"[Train] Device: {device}  |  Folds: {n_splits}  |  Epochs: {epochs}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    best_global_acc = 0.0
    MODEL_DIR.mkdir(exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n── Fold {fold+1}/{n_splits} ──")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Augmentation on training set
        X_tr_aug = np.vstack([X_tr] + [
            np.array([
                _augment_vec(X_tr[i])
                for i in range(len(X_tr))
            ]) for _ in range(2)
        ])
        y_tr_aug = np.tile(y_tr, 3)

        # Dataloaders
        tr_ds  = TensorDataset(torch.tensor(X_tr_aug).float(), torch.tensor(y_tr_aug).long())
        val_ds = TensorDataset(torch.tensor(X_val).float(),    torch.tensor(y_val).long())
        tr_ld  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=batch_size)

        model    = NitrogenCNN(input_dim=X.shape[1]).to(device)
        optim    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched    = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        crit     = nn.BCELoss()
        stopper  = EarlyStopping(patience=20)

        best_val_acc = 0.0
        history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

        for ep in range(epochs):
            tr_loss, tr_acc = train_one_epoch(model, tr_ld, optim, crit, device)
            vl_loss, vl_acc, _, _ = evaluate(model, val_ld, crit, device)
            sched.step()
            history["train_loss"].append(round(tr_loss, 4))
            history["val_loss"].append(round(vl_loss, 4))
            history["train_acc"].append(round(tr_acc * 100, 2))
            history["val_acc"].append(round(vl_acc * 100, 2))

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                # Save fold best
                torch.save(model.state_dict(), MODEL_DIR / f"fold_{fold+1}_best.pth")

            if stopper(vl_loss):
                print(f"   Early stop at epoch {ep+1}")
                break

        fold_results.append({
            "fold": fold + 1,
            "val_accuracy": round(best_val_acc * 100, 2),
            "history": history,
        })
        print(f"   Fold {fold+1} best val acc: {best_val_acc*100:.2f}%")

        # Track global best
        if best_val_acc > best_global_acc:
            best_global_acc = best_val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"   ✅ New global best: {best_global_acc*100:.2f}%")

    accs = [r["val_accuracy"] for r in fold_results]
    summary = {
        "mean_accuracy": round(np.mean(accs), 2),
        "std_accuracy":  round(np.std(accs), 2),
        "folds": fold_results,
    }
    with open(KFOLD_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[K-Fold] Mean accuracy: {summary['mean_accuracy']}% ± {summary['std_accuracy']}%")
    return summary


def _augment_vec(x: np.ndarray) -> np.ndarray:
    rng = np.random
    x = x + rng.normal(0, 0.012, x.shape)
    x = x * rng.uniform(0.94, 1.06)
    mask = rng.rand(len(x)) < 0.04
    x[mask] = 0.0
    return np.clip(x, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model(input_dim: int = 50) -> NitrogenCNN:
    """Load trained model from disk."""
    model = NitrogenCNN(input_dim=input_dim)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(model: NitrogenCNN, x_pca: np.ndarray) -> float:
    """
    Run inference.

    Args:
        model: loaded NitrogenCNN
        x_pca: (50,) PCA feature vector

    Returns:
        score: float in [0,1]  (>0.5 = deficient)
    """
    with torch.no_grad():
        t = torch.tensor(x_pca).float().unsqueeze(0)
        score = model(t).item()
    return score