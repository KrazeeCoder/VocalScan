from __future__ import annotations

import csv
import io
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


def _read_csv(path: str) -> List[List[str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


@dataclass
class SpeechConfig:
    csv_path: str
    label_col: str = "status"
    drop_cols: Sequence[str] = ("name",)


class SpeechFeatureDataset(Dataset):
    """Loads UCI PD voice features csv.

    Returns (x, y) where x is float32 tensor and y is class index (0/1).
    """

    def __init__(self, cfg: SpeechConfig):
        rows = _read_csv(cfg.csv_path)
        header = rows[0]
        data = rows[1:]
        col_idx = {c: i for i, c in enumerate(header)}
        y_idx = col_idx[cfg.label_col]
        drop_idx = {col_idx[c] for c in cfg.drop_cols if c in col_idx}
        x_cols = [i for i in range(len(header)) if i not in drop_idx and i != y_idx]

        feats: List[List[float]] = []
        labels: List[int] = []
        for r in data:
            try:
                labels.append(int(float(r[y_idx])))
                feats.append([float(r[i]) for i in x_cols])
            except Exception:
                continue

        x = np.asarray(feats, dtype=np.float32)
        # simple standardization
        x_mean = x.mean(axis=0, keepdims=True)
        x_std = x.std(axis=0, keepdims=True) + 1e-6
        self.x = (x - x_mean) / x_std
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])


@dataclass
class SpiralConfig:
    root_dir: str  # directory containing .txt coordinate files
    img_size: Tuple[int, int] = (224, 224)
    line_width: int = 3
    norm: bool = True


def _render_spiral_txt_to_image(
    txt_path: str, img_size: Tuple[int, int] = (224, 224), line_width: int = 3
) -> Image.Image:
    # Files are semicolon-separated: X;Y;Z;Pressure;GripAngle;Timestamp;TestID
    xs: List[float] = []
    ys: List[float] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except Exception:
                continue
            xs.append(x)
            ys.append(y)

    if len(xs) < 5:
        return Image.new("L", img_size, color=255)

    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    # normalize to 0..1
    min_x, max_x = xs_arr.min(), xs_arr.max()
    min_y, max_y = ys_arr.min(), ys_arr.max()
    xs_arr = (xs_arr - min_x) / max(1e-6, (max_x - min_x))
    ys_arr = (ys_arr - min_y) / max(1e-6, (max_y - min_y))
    w, h = img_size
    pts = list(zip((xs_arr * (w - 1)).tolist(), (ys_arr * (h - 1)).tolist()))
    img = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)
    for i in range(1, len(pts)):
        draw.line([pts[i - 1], pts[i]], fill=0, width=line_width)
    return img


class SpiralDrawingDataset(Dataset):
    """Renders spiral .txt coordinate files into grayscale images on-the-fly.

    Label is inferred from parent directory name if it contains "parkinson" or "control".
    """

    def __init__(self, cfg: SpiralConfig):
        self.cfg = cfg
        self.items: List[Tuple[str, int]] = []
        for root, _dirs, files in os.walk(cfg.root_dir):
            label = 0
            root_l = root.lower()
            if "parkinson" in root_l:
                label = 1
            elif "control" in root_l:
                label = 0
            for fn in files:
                if fn.lower().endswith(".txt"):
                    self.items.append((os.path.join(root, fn), label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[idx]
        img = _render_spiral_txt_to_image(path, self.cfg.img_size, self.cfg.line_width)
        # To tensor [0,1]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        x = torch.from_numpy(arr)
        y = torch.tensor(label, dtype=torch.long)
        return x, y



# ------------------------- New PNG-based spiral dataset -------------------------
@dataclass
class SpiralImageConfig:
    root_dir: str  # directory containing "Dynamic Spiral Test" and/or "Static Spiral Test" subfolders
    img_size: Tuple[int, int] = (224, 224)
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


class SpiralImageDataset(Dataset):
    """Loads spiral PNGs (grayscale) from spiral_data folders.

    Directory layout expected (case-insensitive match):
      root_dir/
        Dynamic Spiral Test/*.png
        Static Spiral Test/*.png

    This dataset is designed primarily for one-class learning (PD-positive only).
    Labels are set to 1 for all items for compatibility but typically ignored.
    """

    def __init__(self, cfg: SpiralImageConfig):
        self.cfg = cfg
        self.items: List[str] = []
        for root, _dirs, files in os.walk(cfg.root_dir):
            for fn in files:
                if fn.lower().endswith(".png"):
                    self.items.append(os.path.join(root, fn))
        if not self.items:
            raise FileNotFoundError(f"No PNG images found under {cfg.root_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        w, h = self.cfg.img_size
        if img.size != (w, h):
            img = img.resize((w, h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # simple normalization to roughly zero-mean unit-var if desired
        if self.cfg.normalize_std > 0:
            arr = (arr - self.cfg.normalize_mean) / self.cfg.normalize_std
        arr = np.expand_dims(arr, 0)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.items[idx]
        x = self._load_image(path)
        y = torch.tensor(1, dtype=torch.long)  # PD-positive placeholder
        return x, y
