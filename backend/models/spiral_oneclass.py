from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as tvT

from .encoders import SpiralCNN


class OneClassSpiralModel(nn.Module):
    """Encoder-only model for Deep SVDD-style one-class learning on spiral images."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.encoder = SpiralCNN(embed_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


@dataclass
class OneClassTrainConfig:
    root_dir: str
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_ckpt: Optional[str] = None
    embed_dim: int = 128


@torch.no_grad()
def _compute_center(model: OneClassSpiralModel, loader: DataLoader, device: str) -> torch.Tensor:
    model.eval()
    total = None
    n = 0
    for xb, _ in loader:
        xb = xb.to(device)
        z = model(xb)
        if total is None:
            total = z.sum(dim=0)
        else:
            total += z.sum(dim=0)
        n += z.size(0)
    if total is None or n == 0:
        return torch.zeros((model.encoder.proj[1].out_features,), device=device)
    c = total / float(n)
    return c.detach()


@torch.no_grad()
def _collect_distance_stats(model: OneClassSpiralModel, center: torch.Tensor, loader: DataLoader, device: str):
    model.eval()
    dists: List[float] = []
    for xb, _ in loader:
        xb = xb.to(device)
        z = model(xb)
        dist2 = torch.sum((z - center) ** 2, dim=1)
        dists.extend(torch.sqrt(dist2).detach().cpu().numpy().tolist())
    dists_np = np.asarray(dists, dtype=np.float32)
    mu = float(np.mean(dists_np)) if dists_np.size > 0 else 0.0
    sigma = float(np.std(dists_np)) + 1e-6
    return mu, sigma


def _oneclass_augmentations() -> tvT.Compose:
    # Lightweight spatial + photometric augmentations; inputs are 1xHxW tensors [0,1]
    return tvT.Compose([
        tvT.RandomApply([tvT.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05))], p=0.7),
        tvT.RandomHorizontalFlip(p=0.4),
        tvT.RandomVerticalFlip(p=0.1),
        tvT.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=1.0),
    ])


def _apply_aug_batch(xb: torch.Tensor, aug: tvT.Compose) -> torch.Tensor:
    imgs = []
    for i in range(xb.size(0)):
        imgs.append(aug(xb[i]))
    return torch.stack(imgs, dim=0)


def train_oneclass_spiral(
    ds,  # dataset providing (x, _)
    cfg: OneClassTrainConfig,
):
    # Random 80/10/10 split as we have one-class data only
    n = len(ds)
    n_val = max(1, int(0.1 * n))
    n_test = max(1, int(0.1 * n))
    n_train = max(1, n - n_val - n_test)
    ds_train, ds_val, ds_test = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    bs = int(cfg.batch_size)
    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=bs)
    test_loader = DataLoader(ds_test, batch_size=bs)

    model = OneClassSpiralModel(embed_dim=cfg.embed_dim).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    # Initialize center using an initial forward pass over training data
    center = _compute_center(model, train_loader, cfg.device)
    center.requires_grad_(False)

    aug = _oneclass_augmentations()

    best_val = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, _ in train_loader:
            xb = _apply_aug_batch(xb, aug)
            xb = xb.to(cfg.device)
            z = model(xb)
            loss = torch.mean(torch.sum((z - center) ** 2, dim=1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach().cpu()) * xb.size(0)
            total_n += xb.size(0)

        train_loss = total_loss / max(1, total_n)
        # recompute center every few epochs for stability
        if (epoch + 1) % 5 == 0:
            center = _compute_center(model, train_loader, cfg.device)

        # Validation loss = mean distance to center
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_n = 0
            for xb, _ in val_loader:
                xb = xb.to(cfg.device)
                z = model(xb)
                d = torch.mean(torch.sum((z - center) ** 2, dim=1))
                val_total += float(d.detach().cpu()) * xb.size(0)
                val_n += xb.size(0)
            val_loss = val_total / max(1, val_n)

        scheduler.step(val_loss)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} lr={lr_now:.2e}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test loss (distance to center)
    model.eval()
    with torch.no_grad():
        test_total = 0.0
        test_n = 0
        for xb, _ in test_loader:
            xb = xb.to(cfg.device)
            z = model(xb)
            d = torch.mean(torch.sum((z - center) ** 2, dim=1))
            test_total += float(d.detach().cpu()) * xb.size(0)
            test_n += xb.size(0)
        test_loss = test_total / max(1, test_n)
    print(f"test: distance_loss={test_loss:.4f}")

    # Collect distance stats for probability mapping
    mu, sigma = _collect_distance_stats(model, center, train_loader, cfg.device)

    if cfg.save_ckpt:
        ckpt = {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "center": center.detach().cpu(),
            "dist_mean": mu,
            "dist_std": sigma,
            "embed_dim": cfg.embed_dim,
            "img_size": cfg.img_size,
        }
        torch.save(ckpt, cfg.save_ckpt)
        print(f"saved one-class spiral checkpoint to {cfg.save_ckpt}")

    return model, center, mu, sigma


class SpiralOneClassPredictor:
    """Loads one-class spiral checkpoint and produces PD probability from an image tensor/array.

    Probability mapping: p(PD) = sigmoid(-(d - mu) / (sigma + 1e-6)) where d is distance to center.
    """

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        s = torch.load(ckpt_path, map_location="cpu")
        embed_dim = int(s.get("embed_dim", 128))
        self.model = OneClassSpiralModel(embed_dim=embed_dim).to(device)
        self.model.load_state_dict(s["model_state"])
        self.model.eval()
        self.center = s["center"].to(device)
        self.mu = float(s.get("dist_mean", 0.0))
        # Floor sigma to avoid overconfident outputs; clamp to reasonable range
        self.sigma = max(0.05, float(s.get("dist_std", 1.0)))
        self.img_size = tuple(s.get("img_size", (224, 224)))
        # Optional temperature for sigmoid mapping (slightly >1 to desaturate)
        self.temp = float(max(0.5, min(2.5, s.get("prob_temperature", 1.3))))

    @torch.no_grad()
    def predict_proba(self, img: np.ndarray | torch.Tensor) -> float:
        # Accepts either HxW (grayscale) or 1xHxW arrays in [0,1] or [0,255]
        if isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 2:
                arr = arr[None, ...]
            x = torch.from_numpy(arr).unsqueeze(0)  # 1x1xH xW
        else:
            x = img.unsqueeze(0) if img.ndim == 3 else img
        x = x.to(self.device)
        z = self.model(x)
        d = torch.sqrt(torch.sum((z - self.center) ** 2, dim=1))  # shape (1,)
        # Desaturated mapping: p = sigmoid(-(d - mu)/(sigma * temp))
        denom = max(1e-6, self.sigma * self.temp)
        p = torch.sigmoid(-(d - self.mu) / denom)
        return float(p.item())


