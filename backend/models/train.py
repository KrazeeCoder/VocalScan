from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms as tvT

from .datasets import (
    SpeechConfig,
    SpeechFeatureDataset,
    SpiralConfig,
    SpiralDrawingDataset,
    SpiralImageConfig,
    SpiralImageDataset,
)
from .fusion_model import MultimodalPDModel
from .spiral_oneclass import OneClassTrainConfig, train_oneclass_spiral


@dataclass
class TrainConfig:
    speech_csv: Optional[str]
    spiral_root: Optional[str]
    spiral_img_root: Optional[str] = None  # PNG-based spiral dataset (PD-only)
    demo_dim: int = 4  # example: age, gender_onehot(2), family_history
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    # modality-specific optional overrides
    speech_epochs: Optional[int] = None
    spiral_epochs: Optional[int] = None
    speech_lr: Optional[float] = None
    spiral_lr: Optional[float] = None
    speech_batch_size: Optional[int] = None
    spiral_batch_size: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_ckpt: Optional[str] = None


def _train_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    mode: str,
    ce: nn.Module,
    grad_clip: float = 5.0,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if mode == "speech":
            logits = model(None, xb, None)
        elif mode == "spiral":
            # small Gaussian noise augmentation for robustness
            noise = torch.randn_like(xb) * 0.02
            xb = torch.clamp(xb + noise, 0.0, 1.0)
            logits = model(None, None, xb)
        else:
            raise ValueError("mode must be 'speech' or 'spiral'")
        loss = ce(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu()) * yb.size(0)
        n += yb.size(0)
    return total / max(1, n)


@torch.no_grad()
def _evaluate(loader: DataLoader, model: nn.Module, device: str, mode: str) -> tuple[float, float]:
    """Returns (loss, accuracy). mode in {"speech", "spiral"}."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if mode == "speech":
            logits = model(None, xb, None)
        elif mode == "spiral":
            logits = model(None, None, xb)
        else:
            raise ValueError("mode must be 'speech' or 'spiral'")
        total_loss += float(ce(logits, yb).detach().cpu())
        preds = torch.argmax(logits, dim=-1)
        correct += int((preds == yb).sum().item())
        total += yb.size(0)
    loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return loss, acc


def _get_labels_for_speech(ds: SpeechFeatureDataset) -> np.ndarray:
    return np.asarray(ds.y, dtype=np.int64)


def _get_labels_for_spiral(ds: SpiralDrawingDataset) -> np.ndarray:
    return np.asarray([lbl for _p, lbl in ds.items], dtype=np.int64)


def _stratified_indices(labels: np.ndarray, test_frac: float, val_frac: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    idxs = np.arange(labels.shape[0])
    train_idx = []
    val_idx = []
    test_idx = []
    for c in np.unique(labels):
        c_idxs = idxs[labels == c]
        rng.shuffle(c_idxs)
        n = c_idxs.shape[0]
        n_test = max(1, int(round(test_frac * n)))
        n_val = max(0, int(round(val_frac * (n - n_test))))
        test_idx.extend(c_idxs[:n_test].tolist())
        val_idx.extend(c_idxs[n_test : n_test + n_val].tolist())
        train_idx.extend(c_idxs[n_test + n_val :].tolist())
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def _compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    freq = counts.astype(np.float32)
    inv = 1.0 / np.maximum(freq, 1.0)
    weights = inv / inv.sum() * len(classes)
    # Map to full 2-class tensor [w0, w1]
    w = np.ones((2,), dtype=np.float32)
    for c, wt in zip(classes, weights):
        if c < w.shape[0]:
            w[c] = wt
    return torch.from_numpy(w)


def train_unimodal_speech(cfg: TrainConfig) -> None:
    assert cfg.speech_csv is not None, "speech_csv is required"
    ds = SpeechFeatureDataset(SpeechConfig(cfg.speech_csv))
    num_features = ds[0][0].shape[0]
    model = MultimodalPDModel(num_demo_features=cfg.demo_dim, num_speech_features=num_features)
    model.to(cfg.device)
    # Stratified 70/10/20 (train/val/test)
    labels = _get_labels_for_speech(ds)
    train_idx, val_idx, test_idx = _stratified_indices(labels, test_frac=0.2, val_frac=0.1, seed=42)
    ds_train = torch.utils.data.Subset(ds, train_idx.tolist())
    ds_val = torch.utils.data.Subset(ds, val_idx.tolist())
    ds_test = torch.utils.data.Subset(ds, test_idx.tolist())
    bs = int(cfg.speech_batch_size or cfg.batch_size)
    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=bs)
    test_loader = DataLoader(ds_test, batch_size=bs)

    class_weights = _compute_class_weights(labels).to(cfg.device)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    lr = float(cfg.speech_lr or cfg.lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)

    best_val = float("inf")
    best_state = None

    epochs = int(cfg.speech_epochs or cfg.epochs)
    for epoch in range(epochs):
        train_loss = _train_epoch(train_loader, model, opt, cfg.device, mode="speech", ce=ce)
        val_loss, val_acc = _evaluate(val_loader, model, cfg.device, mode="speech")
        scheduler.step(val_loss)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = _evaluate(test_loader, model, cfg.device, mode="speech")
    print(f"test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    if cfg.save_ckpt:
        os.makedirs(os.path.dirname(cfg.save_ckpt) or ".", exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "num_demo": cfg.demo_dim,
                "num_speech": num_features,
            },
            cfg.save_ckpt,
        )
        print(f"saved checkpoint to {cfg.save_ckpt}")


def train_unimodal_spiral(cfg: TrainConfig) -> None:
    assert cfg.spiral_root is not None, "spiral_root is required"
    ds = SpiralDrawingDataset(SpiralConfig(cfg.spiral_root))
    model = MultimodalPDModel(num_demo_features=cfg.demo_dim, num_speech_features=24)  # placeholder
    model.to(cfg.device)
    # Stratified 70/10/20 (train/val/test)
    labels = _get_labels_for_spiral(ds)
    train_idx, val_idx, test_idx = _stratified_indices(labels, test_frac=0.2, val_frac=0.1, seed=42)
    ds_train = torch.utils.data.Subset(ds, train_idx.tolist())
    ds_val = torch.utils.data.Subset(ds, val_idx.tolist())
    ds_test = torch.utils.data.Subset(ds, test_idx.tolist())
    # Class-balanced sampling for training
    train_labels = labels[train_idx]
    classes, counts = np.unique(train_labels, return_counts=True)
    freq = counts.astype(np.float32)
    inv = 1.0 / np.maximum(freq, 1.0)
    class_w = {int(c): float(inv[i]) for i, c in enumerate(classes)}
    sample_w = [class_w[int(l)] for l in train_labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    bs = int(cfg.spiral_batch_size or cfg.batch_size)
    train_loader = DataLoader(ds_train, batch_size=bs, sampler=sampler)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size)
    class_weights = _compute_class_weights(labels).to(cfg.device)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    lr = float(cfg.spiral_lr or cfg.lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)

    best_val = float("inf")
    best_state = None

    # Define lightweight spatial augmentations for spiral drawings
    aug = tvT.Compose([
        tvT.RandomApply([tvT.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.95, 1.05))], p=0.8),
        tvT.RandomHorizontalFlip(p=0.5),
        tvT.RandomVerticalFlip(p=0.1),
        tvT.RandomErasing(p=0.3, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=1.0),
    ])

    epochs = int(cfg.spiral_epochs or cfg.epochs)
    for epoch in range(epochs):
        # augment batch in training loop
        def _spiral_batch_with_aug(xb: torch.Tensor) -> torch.Tensor:
            imgs = []
            for i in range(xb.size(0)):
                imgs.append(aug(xb[i]))
            return torch.stack(imgs, dim=0)

        # monkey-patch _train_epoch call by pre-augmenting batches via closure
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = _spiral_batch_with_aug(xb)
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            logits = model(None, None, xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * yb.size(0)
            n += yb.size(0)
        train_loss = total / max(1, n)
        val_loss, val_acc = _evaluate(val_loader, model, cfg.device, mode="spiral")
        scheduler.step(val_loss)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = _evaluate(test_loader, model, cfg.device, mode="spiral")
    print(f"test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    if cfg.save_ckpt:
        os.makedirs(os.path.dirname(cfg.save_ckpt) or ".", exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "num_demo": cfg.demo_dim,
                "num_speech": 24,
            },
            cfg.save_ckpt,
        )
        print(f"saved checkpoint to {cfg.save_ckpt}")


def _autodetect_and_train(epochs: int, batch_size: int, lr: float, demo_dim: int, device: str) -> None:
    """Zero-config training: detect datasets, train what is available, save under runs/."""
    speech_csv_default = os.path.join("ml_files", "datasets", "voice_recordings", "parkinsons.data")
    spiral_root_default = os.path.join("ml_files", "datasets", "spiral_drawings")
    spiral_img_root_default = os.path.join("ml_files", "datasets", "spiral_data")
    hw_dataset = os.path.join(spiral_root_default, "hw_dataset")
    new_dataset = os.path.join(spiral_root_default, "new_dataset")

    trained_any = False

    if os.path.exists(speech_csv_default):
        print(f"[auto] Found speech CSV: {speech_csv_default}")
        cfg = TrainConfig(
            speech_csv=speech_csv_default,
            spiral_root=None,
            demo_dim=demo_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            # Optimal speech hyperparameters
            speech_epochs=25,
            speech_lr=5e-4,
            speech_batch_size=64,
            device=device,
            save_ckpt=os.path.join("runs", "speech.pt"),
        )
        train_unimodal_speech(cfg)
        trained_any = True
    else:
        print("[auto] Speech CSV not found. Skipping speech training.")

    def _has_spiral_txt(root: str) -> bool:
        if not os.path.isdir(root):
            return False
        for r, _d, files in os.walk(root):
            if any(fn.lower().endswith(".txt") for fn in files):
                return True
        return False

    if _has_spiral_txt(spiral_root_default):
        found_parts = []
        if os.path.isdir(hw_dataset):
            found_parts.append("hw_dataset")
        if os.path.isdir(new_dataset):
            found_parts.append("new_dataset")
        parts = ", ".join(found_parts) if found_parts else "mixed"
        print(f"[auto] Found spiral dataset root: {spiral_root_default} (parts: {parts})")
        cfg = TrainConfig(
            speech_csv=None,
            spiral_root=spiral_root_default,
            demo_dim=demo_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            # Optimal spiral hyperparameters
            spiral_epochs=40,
            spiral_lr=2e-3,
            spiral_batch_size=16,
            device=device,
            save_ckpt=os.path.join("runs", "spiral.pt"),
        )
        train_unimodal_spiral(cfg)
        trained_any = True
    else:
        print("[auto] Spiral dataset not found. Skipping spiral training.")

    # PNG-based PD-only spiral one-class training
    if os.path.isdir(spiral_img_root_default):
        print(f"[auto] Found spiral PNG root: {spiral_img_root_default}")
        ds = SpiralImageDataset(SpiralImageConfig(spiral_img_root_default))
        ocfg = OneClassTrainConfig(
            root_dir=spiral_img_root_default,
            batch_size=32,
            epochs=50,
            lr=2e-3,
            device=device,
            save_ckpt=os.path.join("runs", "spiral_oneclass.pt"),
        )
        # Train using in-memory dataset to honor custom transforms inside trainer
        train_oneclass_spiral(ds, ocfg)
    else:
        print("[auto] Spiral PNG root not found. Skipping one-class spiral training.")

    if not trained_any:
        raise SystemExit(
            "No datasets detected. Place speech CSV at ml_files/datasets/voice_recordings/parkinsons.data "
            "and/or spiral .txt files under ml_files/datasets/spiral_drawings/."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PD model (unimodal speech or spiral)")
    parser.add_argument("--speech_csv", type=str, default=None, help="Path to parkinsons.data-style CSV")
    parser.add_argument("--spiral_root", type=str, default=None, help="Root folder containing spiral .txt files")
    parser.add_argument("--spiral_img_root", type=str, default=None, help="Root folder containing spiral PNGs (PD-only)")
    parser.add_argument("--demo_dim", type=int, default=4, help="Number of demo features (kept for architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Default epochs (overridden by modality-specific if provided)")
    parser.add_argument("--batch_size", type=int, default=32, help="Default batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Default learning rate")
    # Modality-specific overrides
    parser.add_argument("--speech_epochs", type=int, default=None)
    parser.add_argument("--spiral_epochs", type=int, default=None)
    parser.add_argument("--speech_lr", type=float, default=None)
    parser.add_argument("--spiral_lr", type=float, default=None)
    parser.add_argument("--speech_batch_size", type=int, default=None)
    parser.add_argument("--spiral_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; defaults to auto-detect")
    parser.add_argument("--save_ckpt", type=str, default=None, help="Path to save checkpoint .pt")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # If no dataset flags provided, auto-detect and train what is available
    if args.speech_csv is None and args.spiral_root is None and args.spiral_img_root is None:
        _autodetect_and_train(args.epochs, args.batch_size, args.lr, args.demo_dim, device)
        return

    cfg = TrainConfig(
        speech_csv=args.speech_csv,
        spiral_root=args.spiral_root,
        spiral_img_root=args.spiral_img_root,
        demo_dim=args.demo_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        speech_epochs=args.speech_epochs,
        spiral_epochs=args.spiral_epochs,
        speech_lr=args.speech_lr,
        spiral_lr=args.spiral_lr,
        speech_batch_size=args.speech_batch_size,
        spiral_batch_size=args.spiral_batch_size,
        device=device,
        save_ckpt=args.save_ckpt,
    )

    # If provided, train each modality sequentially
    if cfg.speech_csv:
        train_unimodal_speech(cfg)
    if cfg.spiral_root:
        # new cfg for spiral save path if unspecified
        if cfg.save_ckpt is None:
            cfg = TrainConfig(
                speech_csv=None,
                spiral_root=args.spiral_root,
                demo_dim=args.demo_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                save_ckpt=None,
            )
        train_unimodal_spiral(cfg)
    if cfg.spiral_img_root:
        ds = SpiralImageDataset(SpiralImageConfig(cfg.spiral_img_root))
        ocfg = OneClassTrainConfig(
            root_dir=cfg.spiral_img_root,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=args.lr,
            device=device,
            save_ckpt=args.save_ckpt or os.path.join("runs", "spiral_oneclass.pt"),
        )
        train_oneclass_spiral(ds, ocfg)


if __name__ == "__main__":
    main()


