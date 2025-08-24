from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .fusion_model import MultimodalPDModel
from .spiral_oneclass import SpiralOneClassPredictor


def _to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(x.astype(np.float32))
    return t.to(device)


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class WeightedFusion:
    """Linear-probability fusion: p = w_speech * p_speech + w_spiral * p_spiral.

    Weights are normalized to sum to 1 if both modalities are provided.
    If only one modality is provided, its probability is returned.
    """

    w_speech: float = 0.5
    w_spiral: float = 0.5

    def fuse(self, p_speech: Optional[np.ndarray], p_spiral: Optional[np.ndarray]) -> np.ndarray:
        if p_speech is None and p_spiral is None:
            raise ValueError("At least one modality probability must be provided")
        if p_speech is None:
            return p_spiral
        if p_spiral is None:
            return p_speech
        w_sum = max(1e-6, float(self.w_speech + self.w_spiral))
        ws = self.w_speech / w_sum
        wp = self.w_spiral / w_sum
        return ws * p_speech + wp * p_spiral


class LateFusionPredictor:
    """Runs independent unimodal models and fuses their probabilities.

    - Load checkpoints saved by backend.models.train (speech-only and spiral-only runs)
    - Predict with each modality independently
    - Apply simple weighted-probability fusion
    """

    def __init__(
        self,
        speech_ckpt: Optional[str] = None,
        spiral_ckpt: Optional[str] = None,
        spiral_oneclass_ckpt: Optional[str] = None,
        device: Optional[str] = None,
        fusion: Optional[WeightedFusion] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fusion = fusion or WeightedFusion()

        self.speech_model: Optional[MultimodalPDModel] = None
        self.spiral_model: Optional[MultimodalPDModel] = None
        self.spiral_oneclass: Optional[SpiralOneClassPredictor] = None

        if speech_ckpt is not None:
            s = torch.load(speech_ckpt, map_location="cpu")
            self.speech_model = MultimodalPDModel(
                num_demo_features=int(s.get("num_demo", 4)),
                num_speech_features=int(s["num_speech"]),
            ).to(self.device)
            self.speech_model.load_state_dict(s["model_state"])
            self.speech_model.eval()

        if spiral_ckpt is not None:
            sp = torch.load(spiral_ckpt, map_location="cpu")
            self.spiral_model = MultimodalPDModel(
                num_demo_features=int(sp.get("num_demo", 4)),
                num_speech_features=int(sp.get("num_speech", 24)),
            ).to(self.device)
            self.spiral_model.load_state_dict(sp["model_state"])
            self.spiral_model.eval()

        if spiral_oneclass_ckpt is not None:
            self.spiral_oneclass = SpiralOneClassPredictor(spiral_oneclass_ckpt, device=self.device)

        if self.speech_model is None and self.spiral_model is None and self.spiral_oneclass is None:
            raise ValueError("Provide at least one of speech_ckpt, spiral_ckpt, or spiral_oneclass_ckpt")

    @torch.no_grad()
    def predict(
        self,
        speech_vec: Optional[np.ndarray] = None,  # shape (1, D)
        spiral_img: Optional[np.ndarray] = None,  # shape (1, 1, H, W)
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        p_speech: Optional[np.ndarray] = None
        p_spiral: Optional[np.ndarray] = None

        if self.speech_model is not None and speech_vec is not None:
            x = _to_tensor(speech_vec, self.device)
            logits = self.speech_model(None, x, None)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            p_speech = p

        if self.spiral_model is not None and spiral_img is not None:
            x = _to_tensor(spiral_img, self.device)
            logits = self.spiral_model(None, None, x)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            p_spiral = p

        # One-class spiral probability (scalar p(PD)); convert to 2-class probs [p(control), p(PD)]
        if self.spiral_oneclass is not None and spiral_img is not None:
            arr = spiral_img
            if isinstance(arr, np.ndarray):
                if arr.ndim == 4:  # (1,1,H,W)
                    arr = arr[0]
                elif arr.ndim == 3:  # (1,H,W)
                    pass
                else:
                    raise ValueError("spiral_img must be (1,1,H,W) or (1,H,W) for one-class predictor")
            p_pd = float(self.spiral_oneclass.predict_proba(arr))
            p2 = np.asarray([[1.0 - p_pd, p_pd]], dtype=np.float32)
            p_spiral = p2 if p_spiral is None else p_spiral

        fused = self.fusion.fuse(p_speech, p_spiral)
        return p_speech, p_spiral, fused


