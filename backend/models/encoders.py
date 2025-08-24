from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class DemographicsMLP(nn.Module):
    """Simple MLP for tabular demographics and clinical features.

    Input: (batch, num_features)
    Output: (batch, embed_dim)
    """

    def __init__(self, num_features: int, embed_dim: int = 64):
        super().__init__()
        hidden = max(64, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpeechMLP(nn.Module):
    """Encoder for precomputed speech features (e.g., UCI PD, pd_speech_features.csv).

    Input: (batch, num_features)
    Output: (batch, embed_dim)
    """

    def __init__(self, num_features: int, embed_dim: int = 128):
        super().__init__()
        hidden = max(128, embed_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SpiralCNN(nn.Module):
    """Lightweight CNN for spiral drawings rendered as grayscale images.

    Input: (batch, 1, H, W)
    Output: (batch, embed_dim)
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.proj(feats)


class FusionHead(nn.Module):
    """Fuse available modality embeddings via gated summation; supports missing modalities.

    Each modality embedding is projected to a common fusion dimension, gated with a scalar
    in [0,1], and summed (optionally normalized). This avoids fixed concatenation sizes and
    works for unimodal (e.g., voice-only) as well as full multimodal inputs.
    """

    def __init__(
        self,
        embed_dims: Tuple[int, int, int] = (64, 128, 128),
        num_classes: int = 2,
        fusion_dim: int = 128,
        normalize_sum: bool = True,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.normalize_sum = normalize_sum

        # Per-modality projectors and gates (demo, speech, spiral)
        d_e, s_e, sp_e = embed_dims
        self.proj_demo = nn.Linear(d_e, fusion_dim)
        self.proj_speech = nn.Linear(s_e, fusion_dim)
        self.proj_spiral = nn.Linear(sp_e, fusion_dim)

        def gate_block() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1),
                nn.Sigmoid(),
            )

        self.gate_demo = gate_block()
        self.gate_speech = gate_block()
        self.gate_spiral = gate_block()

        self.cls = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        demo: torch.Tensor | None,
        speech: torch.Tensor | None,
        spiral: torch.Tensor | None,
    ) -> torch.Tensor:
        fused = None
        count = 0

        if demo is not None:
            z = self.proj_demo(demo)
            g = self.gate_demo(z)
            fused = (z * g) if fused is None else (fused + z * g)
            count += 1

        if speech is not None:
            z = self.proj_speech(speech)
            g = self.gate_speech(z)
            fused = (z * g) if fused is None else (fused + z * g)
            count += 1

        if spiral is not None:
            z = self.proj_spiral(spiral)
            g = self.gate_spiral(z)
            fused = (z * g) if fused is None else (fused + z * g)
            count += 1

        if fused is None or count == 0:
            raise ValueError("All modalities are missing")

        if self.normalize_sum and count > 1:
            fused = fused / float(count)

        return self.cls(fused)


