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
    """Infer endpoint: verifies Firebase ID token, computes placeholder scores, writes Firestore record.

    Firestore schema:
      users/{uid}
        - lastRecordingAt
        - voiceRecordingPaths: array<string>
        voiceRecordings/{recordId}
          - storagePath, createdAt, durationSec, sampleRate, scores, confidence, riskLevel, modelVersion
    """
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

    # Persist to Firestore in the new schema under users/{uid}
    db = firestore.client()

    # Ensure user doc exists and update aggregates
    user_ref = db.collection("users").document(uid)
    user_ref.set(
        {
            "createdAt": firestore.SERVER_TIMESTAMP,
            "lastRecordingAt": firestore.SERVER_TIMESTAMP,
            # Keep an easy-to-access array of storage filepaths
            "voiceRecordingPaths": firestore.ArrayUnion([storage_path]),
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    # Write detailed per-record doc
    rec_ref = user_ref.collection("voiceRecordings").document(record_id)
    rec_ref.set(
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
            "status": "analyzed",
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


@infer_bp.post("/spiral/infer")
def spiral_infer():
    """Accepts a spiral drawing image, computes placeholder scores, and writes Firestore record.

    Expected multipart/form-data keys:
      - file: image/png or image/jpeg
      - drawingId: optional client-generated id

    Firestore schema under users/{uid}:
      spiralDrawings/{drawingId}
        - storagePath, createdAt, scores, confidence, riskLevel, modelVersion
      Also updates users/{uid}.spiralDrawingPaths array.
    """
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    file_storage = request.files["file"]
    image_bytes = file_storage.read() or b""
    drawing_id_client = (request.form.get("drawingId") or "").strip()

    # Reuse placeholder scoring based on content
    scores, confidence, risk_level = _compute_placeholder_scores(image_bytes)

    # Create or use provided drawing id
    drawing_id = drawing_id_client or time.strftime("spiral_%Y%m%d_%H%M%S")
    storage_path = f"spirals/{uid}/{drawing_id}.png"

    db = firestore.client()
    user_ref = db.collection("users").document(uid)
    user_ref.set(
        {
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "lastSpiralAt": firestore.SERVER_TIMESTAMP,
            "spiralDrawingPaths": firestore.ArrayUnion([storage_path]),
        },
        merge=True,
    )

    spiral_ref = user_ref.collection("spiralDrawings").document(drawing_id)
    spiral_ref.set(
        {
            "createdAt": firestore.SERVER_TIMESTAMP,
            "storagePath": storage_path,
            "modelVersion": "placeholder-v1",
            "scores": scores,
            "confidence": confidence,
            "riskLevel": risk_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "analyzed",
        },
        merge=True,
    )

    return (
        jsonify(
            {
                "drawingId": drawing_id,
                "modelVersion": "placeholder-v1",
                "scores": scores,
                "confidence": confidence,
                "riskLevel": risk_level,
                "storagePath": storage_path,
            }
        ),
        200,
    )


@infer_bp.post("/demographics")
def submit_demographics():
    """Save user demographic information to Firestore."""
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    # Get JSON data from request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No demographic data provided"}), 400

    # Validate required fields (skip validation for welcome flow updates)
    if not data.get("welcomeFlowStarted") and not data.get("welcomeFlowCompleted"):
        required_fields = ["age", "gender", "medicalHistory"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

    # Prepare demographic data
    demographic_data = {
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    # Only add full demographic data if this isn't just a welcome flow update
    if not data.get("welcomeFlowStarted") and not data.get("welcomeFlowCompleted"):
        demographic_data.update({
            "age": data.get("age"),
            "gender": data.get("gender"),
            "medicalHistory": data.get("medicalHistory"),
            "symptoms": data.get("symptoms", []),
            "medications": data.get("medications", []),
            "familyHistory": data.get("familyHistory", []),
            "lifestyle": data.get("lifestyle", {}),
            "contactInfo": data.get("contactInfo", {}),
            "emergencyContact": data.get("emergencyContact", {}),
            "demographicsCompleted": True,
            "demographicsCompletedAt": firestore.SERVER_TIMESTAMP,
        })

    # Handle welcome flow flags
    if "welcomeFlowStarted" in data:
        demographic_data["welcomeFlowStarted"] = data["welcomeFlowStarted"]
    if "welcomeFlowCompleted" in data:
        demographic_data["welcomeFlowCompleted"] = data["welcomeFlowCompleted"]
        if data["welcomeFlowCompleted"]:
            demographic_data["welcomeFlowCompletedAt"] = firestore.SERVER_TIMESTAMP

    # Update user document with demographic information
    db = firestore.client()
    user_ref = db.collection("users").document(uid)
    user_ref.set(demographic_data, merge=True)

    return jsonify({"success": True, "message": "Demographics saved successfully"}), 200


@infer_bp.get("/demographics/status")
def check_demographics_status():
    """Check if user has completed demographics."""
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    db = firestore.client()
    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()
    
    if user_doc.exists:
        user_data = user_doc.to_dict()
        demographics_completed = user_data.get("demographicsCompleted", False)
        welcome_flow_completed = user_data.get("welcomeFlowCompleted", False)
    else:
        demographics_completed = False
        welcome_flow_completed = False

    return jsonify({
        "demographicsCompleted": demographics_completed,
        "welcomeFlowCompleted": welcome_flow_completed
    }), 200


@infer_bp.post("/api/analyze-spiral")
def analyze_spiral():
    """Analyze spiral drawing from assessment flow.
    
    Expected JSON payload:
      - path: array of drawing coordinates with timestamps
      - image: base64 image data
      - timestamp: ISO timestamp
    """
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    spiral_path = data.get("path", [])
    image_data = data.get("image", "")
    timestamp = data.get("timestamp", "")

    if not spiral_path:
        return jsonify({"error": "No drawing path provided"}), 400

    # Generate placeholder analysis based on drawing characteristics
    def analyze_spiral_characteristics(path_data):
        """Compute placeholder scores based on drawing path."""
        if not path_data or len(path_data) < 10:
            return {"tremor": 0.8, "smoothness": 0.2}, 0.3, "HIGH"
        
        # Calculate basic drawing metrics
        total_points = len(path_data)
        
        # Simulate tremor analysis (more points = potentially more tremor)
        tremor_score = min(total_points / 1000.0, 0.9)
        
        # Simulate smoothness (consistent timing = smoother)
        timestamps = [p.get("timestamp", 0) for p in path_data]
        if len(timestamps) > 1:
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 1
            smoothness_score = max(0.1, min(0.9, 1.0 - (tremor_score * 0.5)))
        else:
            smoothness_score = 0.5
        
        scores = {
            "tremor": round(tremor_score, 3),
            "smoothness": round(smoothness_score, 3),
            "coordination": round((smoothness_score + (1.0 - tremor_score)) / 2, 3)
        }
        
        # Determine risk level
        max_risk_score = max(tremor_score, 1.0 - smoothness_score)
        if max_risk_score < 0.33:
            risk = "LOW"
        elif max_risk_score < 0.66:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        confidence = round(0.7 + (smoothness_score * 0.2), 2)
        return scores, confidence, risk

    scores, confidence, risk_level = analyze_spiral_characteristics(spiral_path)

    # Generate unique drawing ID
    drawing_id = f"spiral_{int(time.time() * 1000)}"

    # Save to Firestore
    db = firestore.client()
    user_ref = db.collection("users").document(uid)
    
    # Update user summary
    user_ref.set({
        "lastSpiralDrawingAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)

    # Save detailed spiral record
    spiral_ref = user_ref.collection("spiralDrawings").document(drawing_id)
    spiral_ref.set({
        "createdAt": firestore.SERVER_TIMESTAMP,
        "drawingPath": spiral_path,
        "imageData": image_data[:100],  # Store truncated image data for privacy
        "modelVersion": "placeholder-v1",
        "scores": scores,
        "confidence": confidence,
        "riskLevel": risk_level,
        "timestamp": timestamp,
        "status": "analyzed",
        "totalPoints": len(spiral_path),
    })

    return jsonify({
        "drawingId": drawing_id,
        "modelVersion": "placeholder-v1",
        "scores": scores,
        "confidence": confidence,
        "riskLevel": risk_level,
        "analysis": {
            "tremor": scores["tremor"],
            "smoothness": scores["smoothness"],
            "coordination": scores["coordination"]
        }
    }), 200


