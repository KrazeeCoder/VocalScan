from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .encoders import DemographicsMLP, SpeechMLP, SpiralCNN, FusionHead


class MultimodalPDModel(nn.Module):
    """End-to-end multimodal model with optional modalities.

    - Demographics/clinical tabular features: shape (B, Dd)
    - Speech features (precomputed): shape (B, Ds)
    - Spiral drawing image (grayscale 1xHxW): shape (B, 1, H, W)

    Any modality can be None during forward.
    """

    def __init__(
        self,
        num_demo_features: int,
        num_speech_features: int,
        demo_embed: int = 64,
        speech_embed: int = 128,
        spiral_embed: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.demo_encoder = DemographicsMLP(num_demo_features, demo_embed)
        self.speech_encoder = SpeechMLP(num_speech_features, speech_embed)
        self.spiral_encoder = SpiralCNN(spiral_embed)
        self.head = FusionHead((demo_embed, speech_embed, spiral_embed), num_classes)

    def forward(
        self,
        demo_x: Optional[torch.Tensor],
        speech_x: Optional[torch.Tensor],
        spiral_x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        demo_z = self.demo_encoder(demo_x) if demo_x is not None else None
        speech_z = self.speech_encoder(speech_x) if speech_x is not None else None
        spiral_z = self.spiral_encoder(spiral_x) if spiral_x is not None else None
        return self.head(demo_z, speech_z, spiral_z)


