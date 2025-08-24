from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Tuple

from flask import Blueprint, jsonify, request
from firebase_admin import firestore

from .auth import AuthError, verify_firebase_id_token


infer_bp = Blueprint("infer", __name__)


def _compute_placeholder_scores(audio_bytes: bytes) -> Tuple[dict, float, str]:
    """Deterministic pseudo-scores and risk level based on content hash."""
    if not audio_bytes:
        digest = hashlib.sha256(b"empty").digest()
    else:
        digest = hashlib.sha256(audio_bytes[:65536]).digest()

    respiratory = round((digest[0] / 255.0) * 0.8, 3)
    neurological = round((digest[1] / 255.0) * 0.8, 3)
    scores = {"respiratory": respiratory, "neurological": neurological}

    max_score = max(scores.values())
    if max_score < 0.33:
        risk = "low"
    elif max_score < 0.66:
        risk = "med"
    else:
        risk = "high"

    confidence = round(0.6 + (digest[2] / 255.0) * 0.3, 2)
    return scores, confidence, risk


@infer_bp.post("/infer")
def infer():
    """Infer endpoint: verifies Firebase ID token, computes placeholder scores, writes Firestore record."""
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    # Read inputs
    file_storage = request.files["file"]
    audio_bytes = file_storage.read() or b""

    sample_rate_str = request.form.get("sampleRate", "48000").strip()
    duration_sec_str = request.form.get("durationSec", "10").strip()
    record_id_client = request.form.get("recordId", "").strip()

    try:
        sample_rate = int(float(sample_rate_str))
    except ValueError:
        sample_rate = 48000

    try:
        duration_sec = float(duration_sec_str)
    except ValueError:
        duration_sec = 10.0

    # Create or use provided record id
    record_id = record_id_client or time.strftime("rec_%Y%m%d_%H%M%S")
    storage_path = f"audio/{uid}/{record_id}.webm"

    # Placeholder model inference
    scores, confidence, risk_level = _compute_placeholder_scores(audio_bytes)

    # Persist to Firestore
    db = firestore.client()
    doc_ref = db.collection("tests").document(uid).collection("records").document(record_id)
    doc_ref.set(
        {
            "createdAt": firestore.SERVER_TIMESTAMP,
            "storagePath": storage_path,
            "durationSec": duration_sec,
            "sampleRate": sample_rate,
            "modelVersion": "placeholder-v1",
            "scores": scores,
            "confidence": confidence,
            "riskLevel": risk_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        merge=True,
    )

    return (
        jsonify(
            {
                "recordId": record_id,
                "modelVersion": "placeholder-v1",
                "scores": scores,
                "confidence": confidence,
                "riskLevel": risk_level,
                "storagePath": storage_path,
            }
        ),
        200,
    )


