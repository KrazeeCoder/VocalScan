from datetime import datetime, timezone
from flask import Blueprint, jsonify, request

from .auth_mock import require_firebase_auth  # Use mock auth for development
from .models.placeholder import run_placeholder_inference


infer_bp = Blueprint("infer", __name__)


@infer_bp.post("/predict")
@require_firebase_auth
def predict_route():
    """Main prediction endpoint for VocalScan."""
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

    # Get sample type (voice, cough, breath, sustained, sentence)
    sample_type = request.form.get("sampleType", "voice").lower()
    
    record_id = request.form.get("recordId")
    if not record_id:
        now = datetime.now(timezone.utc)
        record_id = now.strftime("rec_%Y-%m-%d_%H-%M-%S")

    scores, confidence, risk_level, model_version = run_placeholder_inference(
        audio_bytes=audio_bytes, 
        sample_rate=sample_rate, 
        duration_sec=duration_sec,
        sample_type=sample_type
    )

    # Add interpretation based on results
    interpretation = _generate_interpretation(scores, risk_level, sample_type)

    response = {
        "recordId": record_id,
        "modelVersion": model_version,
        "sampleType": sample_type,
        "scores": scores,
        "confidence": confidence,
        "riskLevel": risk_level,
        "interpretation": interpretation,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return jsonify(response), 200


@infer_bp.post("/infer")
@require_firebase_auth
def infer_route():
    """Legacy inference endpoint (backwards compatibility)."""
    return predict_route()


@infer_bp.delete("/delete/<record_id>")
@require_firebase_auth
def delete_record(record_id: str):
    """Delete user's record for privacy."""
    # For now, just return success (no actual storage implemented)
    return jsonify({"message": f"Record {record_id} deleted successfully"}), 200


def _generate_interpretation(scores: dict, risk_level: str, sample_type: str) -> dict:
    """Generate user-friendly interpretation of results."""
    
    respiratory_score = scores.get("respiratory", 0)
    neurological_score = scores.get("neurological", 0)
    
    interpretation = {
        "summary": "",
        "details": [],
        "nextSteps": [],
        "disclaimer": "This is a pattern analysis tool, not a medical diagnosis. Consult healthcare professionals for medical advice."
    }
    
    # Generate summary based on risk level
    if risk_level == "low":
        interpretation["summary"] = "Low likelihood of concerning patterns detected."
        interpretation["nextSteps"] = [
            "Continue regular health monitoring",
            "Consider periodic re-testing if symptoms develop"
        ]
    elif risk_level == "medium":
        interpretation["summary"] = "Some patterns of interest detected. Consider monitoring."
        interpretation["nextSteps"] = [
            "Monitor symptoms and voice changes",
            "Consider consultation with healthcare provider if patterns persist",
            "Retest in a few weeks"
        ]
    else:  # high
        interpretation["summary"] = "Notable patterns detected that may warrant attention."
        interpretation["nextSteps"] = [
            "Consider consultation with a healthcare provider",
            "Document any symptoms or voice changes",
            "Follow up testing may be beneficial"
        ]
    
    # Add specific details based on sample type and scores
    if sample_type in ["cough", "breath"] and respiratory_score > 0.3:
        interpretation["details"].append(
            f"Respiratory analysis shows patterns that may indicate breathing irregularities (score: {respiratory_score:.2f})"
        )
    
    if sample_type in ["voice", "sustained", "sentence"] and neurological_score > 0.3:
        interpretation["details"].append(
            f"Voice analysis shows patterns that may indicate vocal changes (score: {neurological_score:.2f})"
        )
    
    if not interpretation["details"]:
        interpretation["details"].append("Analysis shows patterns within normal ranges.")
    
    return interpretation


