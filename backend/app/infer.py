from datetime import datetime, timezone
from flask import Blueprint, jsonify, request

from .auth import require_firebase_auth
from .models.placeholder import run_placeholder_inference


infer_bp = Blueprint("infer", __name__)


@infer_bp.post("/infer")
@require_firebase_auth
def infer_route():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in form-data"}), 400

    file_storage = request.files["file"]
    audio_bytes = file_storage.read()

    sample_rate_str = request.form.get("sampleRate")
    duration_sec_str = request.form.get("durationSec")
    if not sample_rate_str or not duration_sec_str:
        return jsonify({"error": "Missing 'sampleRate' or 'durationSec'"}), 400

    try:
        sample_rate = int(sample_rate_str)
        duration_sec = float(duration_sec_str)
    except ValueError:
        return jsonify({"error": "Invalid 'sampleRate' or 'durationSec'"}), 400

    record_id = request.form.get("recordId")
    if not record_id:
        now = datetime.now(timezone.utc)
        record_id = now.strftime("rec_%Y-%m-%d_%H-%M-%S")

    scores, confidence, risk_level, model_version = run_placeholder_inference(
        audio_bytes=audio_bytes, sample_rate=sample_rate, duration_sec=duration_sec
    )

    response = {
        "recordId": record_id,
        "modelVersion": model_version,
        "scores": scores,
        "confidence": confidence,
        "riskLevel": risk_level,
    }
    return jsonify(response), 200


