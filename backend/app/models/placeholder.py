from __future__ import annotations

import hashlib
import random
from typing import Dict, Tuple
from .ml_models import get_model


def _seed_rng_from_bytes(seed: bytes) -> random.Random:
    seed_int = int.from_bytes(seed[:8], "big", signed=False)
    return random.Random(seed_int)


def _compute_byte_amplitude_estimate(data: bytes) -> float:
    """Compute a crude amplitude estimate based on byte distribution.

    This is container-agnostic and purely for deterministic placeholder behavior.
    Returns a value in [0, 1].
    """
    if not data:
        return 0.0

    window = data[:200_000]
    total = 0.0
    for b in window:
        total += abs(b - 128) / 128.0
    return min(total / max(len(window), 1), 1.0)


def run_placeholder_inference(
    audio_bytes: bytes, sample_rate: int, duration_sec: float, sample_type: str = "voice"
) -> Tuple[Dict[str, float], float, str, str]:
    """Return ML-based scores for the MVP.

    Returns (scores, confidence, risk_level, model_version).
    """
    try:
        # Use the actual ML model
        model = get_model()
        result = model.predict(audio_bytes, sample_type)
        
        return (
            result["scores"],
            result["confidence"],
            result["risk_level"],
            result["model_version"]
        )
        
    except Exception as e:
        # Fallback to deterministic placeholder if ML fails
        print(f"ML prediction failed, using fallback: {e}")
        
        digest = hashlib.sha256(audio_bytes[:100_000]).digest()
        rng = _seed_rng_from_bytes(digest)
        amplitude = _compute_byte_amplitude_estimate(audio_bytes)

        base = rng.random() * 0.7 + 0.15
        dur_factor = min(duration_sec / 30.0, 1.0) * 0.15
        amp_factor = amplitude * 0.2

        respiratory = max(0.0, min(base + dur_factor + amp_factor + (rng.random() - 0.5) * 0.1, 1.0))
        neurological = max(0.0, min(base + (1 - dur_factor) * 0.1 + amp_factor * 0.5 + (rng.random() - 0.5) * 0.1, 1.0))

        scores: Dict[str, float] = {
            "respiratory": round(respiratory, 2),
            "neurological": round(neurological, 2),
        }

        confidence = round(0.6 + 0.3 * rng.random(), 2)
        top = max(scores.values())
        risk_level = "low" if top < 0.33 else ("medium" if top < 0.66 else "high")
        model_version = "fallback-v1"

        return scores, confidence, risk_level, model_version


