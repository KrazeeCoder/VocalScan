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
    """Linear probability fusion for up to three modalities (speech, spiral, demo).

    If multiple modalities are provided, their weights are renormalized to sum to 1.
    If only one modality is provided, its probability is returned unchanged.
    """

    w_speech: float = 0.4
    w_spiral: float = 0.4
    w_demo: float = 0.2

    def fuse(
        self,
        p_speech: Optional[np.ndarray],
        p_spiral: Optional[np.ndarray],
        p_demo: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        provided = []
        weights = []
        if p_speech is not None:
            provided.append(p_speech)
            weights.append(self.w_speech)
        if p_spiral is not None:
            provided.append(p_spiral)
            weights.append(self.w_spiral)
        if p_demo is not None:
            provided.append(p_demo)
            weights.append(self.w_demo)
        if not provided:
            raise ValueError("At least one modality probability must be provided")
        if len(provided) == 1:
            return provided[0]
        w_sum = max(1e-6, float(sum(weights)))
        weights = [w / w_sum for w in weights]
        fused = np.zeros_like(provided[0], dtype=np.float32)
        for w, p in zip(weights, provided):
            fused = fused + float(w) * p
        return fused


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
        temperature_speech: float = 1.0,
        temperature_spiral: float = 1.5,
        prob_eps: float = 1e-3,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fusion = fusion or WeightedFusion()
        # Calibration params
        self.temp_speech = float(max(1e-3, temperature_speech))
        self.temp_spiral = float(max(1e-3, temperature_spiral))
        self.prob_eps = float(max(0.0, prob_eps))

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
        p_demo: Optional[np.ndarray] = None,      # shape (1, 2)
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        p_speech: Optional[np.ndarray] = None
        p_spiral: Optional[np.ndarray] = None

        if self.speech_model is not None and speech_vec is not None:
            x = _to_tensor(speech_vec, self.device)
            logits = self.speech_model(None, x, None)
            # Temperature-scaled softmax for desaturation
            p = F.softmax(logits / self.temp_speech, dim=-1).cpu().numpy()
            if self.prob_eps > 0:
                p = np.clip(p, self.prob_eps, 1.0 - self.prob_eps)
                p = p / p.sum(axis=1, keepdims=True)
            p_speech = p

        if self.spiral_model is not None and spiral_img is not None:
            x = _to_tensor(spiral_img, self.device)
            logits = self.spiral_model(None, None, x)
            # Temperature-scaled softmax for desaturation
            p = F.softmax(logits / self.temp_spiral, dim=-1).cpu().numpy()
            if self.prob_eps > 0:
                p = np.clip(p, self.prob_eps, 1.0 - self.prob_eps)
                p = p / p.sum(axis=1, keepdims=True)
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

        fused = self.fusion.fuse(p_speech, p_spiral, p_demo)
        return p_speech, p_spiral, p_demo, fused


