from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile
import time
import logging
import shutil
import subprocess
from datetime import datetime, timezone
import json
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from flask import Blueprint, jsonify, request
from firebase_admin import firestore

from .auth import AuthError, verify_firebase_id_token
from ..models.voice_features import UCIPDVectorizer, vectorize_wav, extract_ucipd_from_wav
from ..models.late_fusion import LateFusionPredictor, WeightedFusion


infer_bp = Blueprint("infer", __name__)
logger = logging.getLogger(__name__)


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


def _find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg executable with multiple strategies.

    Priority: env var FFMPEG_PATH > PATH lookup > common Windows install paths.
    """
    candidates = []
    env_path = os.getenv("FFMPEG_PATH")
    if env_path:
        candidates.append(env_path)
    which_path = shutil.which("ffmpeg")
    if which_path:
        candidates.append(which_path)
    # Common Windows install locations
    candidates.extend(
        [
            r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            r"C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
            r"C:\\ffmpeg\\bin\\ffmpeg.exe",
            r"C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe",
        ]
    )
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    # Try bundled ffmpeg from imageio-ffmpeg if available
    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return None


def _parse_speech_features_from_request(req) -> Optional[np.ndarray]:
    """Extract a 1xD float32 array of speech features from JSON or form fields.

    Accepts:
      - JSON body: { "speech_features": [..] }
      - Multipart form: speech_features or speech_features_json as JSON string or comma-separated floats
    """
    try:
        if req.is_json:
            data = req.get_json(silent=True) or {}
            feats = data.get("speech_features")
            if isinstance(feats, list) and len(feats) > 0:
                return np.asarray(feats, dtype=np.float32).reshape(1, -1)
        s = req.form.get("speech_features") or req.form.get("speech_features_json")
        if s:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return np.asarray(parsed, dtype=np.float32).reshape(1, -1)
            except Exception:
                parts = [p for p in s.split(",") if p.strip()]
                if parts:
                    vals = [float(p) for p in parts]
                    return np.asarray(vals, dtype=np.float32).reshape(1, -1)
    except Exception:
        pass
    return None


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

    if (
        "file" not in request.files
        and "audio_wav" not in request.files
        and _parse_speech_features_from_request(request) is None
    ):
        return jsonify({"error": "missing input: provide audio or speech_features"}), 400

    # Read inputs
    file_storage = request.files.get("file")
    audio_bytes = (file_storage.read() if file_storage else b"") or b""
    provided_speech_features: Optional[np.ndarray] = _parse_speech_features_from_request(request)

    # Optional WAV upload (preferred for real model inference)
    wav_upload = request.files.get("audio_wav")
    tmp_wav_path: Optional[str] = None
    if wav_upload is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(wav_upload.read())
                tmp_wav_path = tmp.name
        except Exception:
            tmp_wav_path = None

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

    logger.info(
        "infer: uid=%s record_id=%s audio_bytes=%d sample_rate=%s duration_sec=%s",
        uid,
        record_id,
        len(audio_bytes),
        sample_rate,
        duration_sec,
    )

    # Enforce real speech inference (no placeholders)
    try:
        _ensure_models_loaded()
        if _FUSION_PREDICTOR is None:
            return jsonify({"error": "models_unavailable"}), 503
        if _VECTORIZER is None:
            return jsonify({"error": "vectorizer_unavailable"}), 503

        # If client provides precomputed speech features, use them directly
        if provided_speech_features is not None:
            speech_vec = provided_speech_features
        else:
            # Otherwise, expect audio and vectorize
            if tmp_wav_path is None:
                if not audio_bytes:
                    return jsonify({"error": "no_audio_data"}), 400
                ffmpeg_path = _find_ffmpeg()
                if ffmpeg_path is None:
                    return jsonify({"error": "ffmpeg_not_found"}), 503
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
                        tmp_in.write(audio_bytes)
                        tmp_in_path = tmp_in.name
                    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tmp_out_path = tmp_out.name
                    tmp_out.close()
                    subprocess.run(
                        [
                            ffmpeg_path,
                            "-y",
                            "-i",
                            tmp_in_path,
                            "-vn",
                            "-ar",
                            str(sample_rate),
                            "-ac",
                            "1",
                            "-acodec",
                            "pcm_s16le",
                            "-f",
                            "wav",
                            tmp_out_path,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    tmp_wav_path = tmp_out_path
                finally:
                    try:
                        if 'tmp_in_path' in locals() and os.path.exists(tmp_in_path):
                            os.remove(tmp_in_path)
                    except Exception:
                        pass

            feat_dict, speech_vec = _extract_and_vectorize_wav(tmp_wav_path)
            if speech_vec is None or feat_dict is None:
                logger.error("infer: vectorize_failed for file=%s; ensure praat-parselmouth is installed and WAV valid", tmp_wav_path)
                return jsonify({"error": "vectorize_failed"}), 500

        db = firestore.client()
        user_ref = db.collection("users").document(uid)
        user_doc = user_ref.get()
        profile = user_doc.to_dict() if user_doc.exists else {}
        p_demo = _demographics_prior(profile)

        _p_s, _p_sp, _p_d, fused = _FUSION_PREDICTOR.predict(
            speech_vec=speech_vec, spiral_img=None, p_demo=p_demo
        )
        risk_level, confidence = _risk_from_probs(fused)
        logger.info(
            "infer(real): uid=%s record_id=%s risk=%s confidence=%.3f",
            uid,
            record_id,
            risk_level,
            confidence,
        )
        scores = {"pd": float(fused[0][1])}
    except Exception as e:
        logger.exception("infer: real model inference error: %s", e)
        return jsonify({"error": "inference_failed"}), 500
    finally:
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            try:
                os.remove(tmp_wav_path)
            except Exception:
                pass

    logger.info(
        "infer: result record_id=%s risk=%s confidence=%.2f",
        record_id,
        risk_level,
        confidence,
    )

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
    rec = {
            "createdAt": firestore.SERVER_TIMESTAMP,
            "storagePath": storage_path,
            "durationSec": duration_sec,
            "sampleRate": sample_rate,
            "modelVersion": "speech-v1",
            "scores": scores,
            "confidence": confidence,
            "riskLevel": risk_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "analyzed",
        }
    # Store extracted features if available
    try:
        if 'feat_dict' in locals() and feat_dict is not None:
            rec["features"] = {k: float(v) for k, v in (feat_dict or {}).items()}
    except Exception:
        pass
    rec_ref.set(rec, merge=True)

    return (
        jsonify(
            {
                "recordId": record_id,
                "modelVersion": "speech-v1",
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

    # Enforce real spiral inference (no placeholders)
    scores = None
    confidence = None
    risk_level = None

    # Create or use provided drawing id
    drawing_id = drawing_id_client or time.strftime("spiral_%Y%m%d_%H%M%S")
    storage_path = f"spirals/{uid}/{drawing_id}.png"

    try:
        _ensure_models_loaded()
        if _FUSION_PREDICTOR is None:
            return jsonify({"error": "models_unavailable"}), 503
        if not image_bytes:
            return jsonify({"error": "no_image_data"}), 400
        spiral_img = _image_bytes_to_spiral_tensor(image_bytes)
        if spiral_img is None:
            return jsonify({"error": "invalid_image"}), 400
        p_s, p_sp, p_d, fused = _FUSION_PREDICTOR.predict(
            speech_vec=None, spiral_img=spiral_img, p_demo=None
        )
        risk_level, confidence = _risk_from_probs(fused)
        scores = {"pd": float(fused[0][1])}
        logger.info(
            "spiral_infer(real): uid=%s drawing_id=%s risk=%s confidence=%.3f",
            uid,
            drawing_id,
            risk_level,
            confidence,
        )
    except Exception as e:
        logger.exception("spiral_infer: real model inference error: %s", e)
        return jsonify({"error": "inference_failed"}), 500

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
            "modelVersion": "spiral-v1",
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
                "modelVersion": "spiral-v1",
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


@infer_bp.get("/demographics/data")
def get_demographics_data():
    """Get user demographics data for AI Coach."""
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
        
        if demographics_completed:
            # Return relevant demographics data for AI Coach
            return jsonify({
                "age": user_data.get("age"),
                "gender": user_data.get("gender"),
                "medicalHistory": user_data.get("medicalHistory"),
                "symptoms": user_data.get("symptoms", []),
                "medications": user_data.get("medications", []),
                "familyHistory": user_data.get("familyHistory", []),
                "lifestyle": user_data.get("lifestyle", {}),
                "demographicsCompleted": True
            }), 200
        else:
            return jsonify({
                "demographicsCompleted": False,
                "message": "Demographics not completed"
            }), 200
    else:
        return jsonify({
            "demographicsCompleted": False,
            "message": "User not found"
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

    


# -----------------------------
# Multimodal model inference (speech + spiral + demographics)
# -----------------------------

_FUSION_PREDICTOR: Optional[LateFusionPredictor] = None
_VECTORIZER: Optional[UCIPDVectorizer] = None


def _detect_ckpt(path: str) -> Optional[str]:
    return path if os.path.exists(path) else None


def _ensure_models_loaded() -> None:
    global _FUSION_PREDICTOR, _VECTORIZER
    if _FUSION_PREDICTOR is None:
        runs_dir = os.path.join(os.getcwd(), "runs")
        speech_ckpt = _detect_ckpt(os.path.join(runs_dir, "speech.pt"))
        spiral_ckpt = _detect_ckpt(os.path.join(runs_dir, "spiral.pt"))
        spiral_oneclass_ckpt = _detect_ckpt(os.path.join(runs_dir, "spiral_oneclass.pt"))
        logger.info(
            "_ensure_models_loaded: runs_dir=%s speech_ckpt=%s spiral_ckpt=%s spiral_oneclass_ckpt=%s",
            runs_dir,
            bool(speech_ckpt),
            bool(spiral_ckpt),
            bool(spiral_oneclass_ckpt),
        )
        _FUSION_PREDICTOR = LateFusionPredictor(
            speech_ckpt=speech_ckpt,
            spiral_ckpt=spiral_ckpt,
            spiral_oneclass_ckpt=spiral_oneclass_ckpt,
            fusion=WeightedFusion(w_speech=0.55, w_spiral=0.4, w_demo=0.05),
            temperature_speech=1.7,
            temperature_spiral=2.3,
            prob_eps=0.01,
        )
        logger.info("_ensure_models_loaded: LateFusionPredictor initialized")
    if _VECTORIZER is None:
        csv_default = os.path.join("ml_files", "datasets", "voice_recordings", "parkinsons.data")
        if os.path.exists(csv_default):
            _VECTORIZER = UCIPDVectorizer.from_csv(csv_default)
            logger.info("_ensure_models_loaded: UCIPDVectorizer loaded from %s", csv_default)
        else:
            logger.info("_ensure_models_loaded: UCIPDVectorizer CSV not found at %s", csv_default)


def _load_wav_to_vec(tmp_wav_path: str) -> Optional[np.ndarray]:
    if _VECTORIZER is None:
        return None


def _extract_and_vectorize_wav(tmp_wav_path: str) -> Tuple[Optional[dict], Optional[np.ndarray]]:
    """Extract raw UCI-PD feature dict from WAV, then standardize to model vector.

    Returns (feature_dict, vector) or (None, None) on error.
    """
    if _VECTORIZER is None:
        return None, None
    try:
        feat_dict = extract_ucipd_from_wav(tmp_wav_path)
        vec = _VECTORIZER.transform(feat_dict).astype(np.float32)
        return feat_dict, vec
    except Exception as e:
        logger.exception("_extract_and_vectorize_wav: failed for %s: %s", tmp_wav_path, e)
        return None, None
    try:
        vec = vectorize_wav(tmp_wav_path, _VECTORIZER)
        return vec.astype(np.float32)
    except Exception as e:
        logger.exception("_load_wav_to_vec: failed to vectorize %s: %s", tmp_wav_path, e)
        return None


def _image_bytes_to_spiral_tensor(img_bytes: bytes, size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        if img.size != size:
            img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.reshape(1, 1, size[1], size[0])
        return arr
    except Exception:
        return None


def _dataurl_to_bytes(data_url: str) -> Optional[bytes]:
    try:
        if data_url.startswith("data:"):
            header, b64 = data_url.split(",", 1)
            return base64.b64decode(b64)
        # treat as base64 plain string
        return base64.b64decode(data_url)
    except Exception:
        return None


def _demographics_prior(user_doc: dict) -> Optional[np.ndarray]:
    # Apply demographics prior only when the user completed the questionnaire
    if not user_doc or not bool(user_doc.get("demographicsCompleted")):
        return None
    age = user_doc.get("age")
    gender = (user_doc.get("gender") or "").lower()
    fam = user_doc.get("familyHistory") or []
    try:
        age = float(age) if age is not None else None
    except Exception:
        age = None
    # Conservative baseline; prevalence is low
    p_pd = 0.10
    if age is not None:
        if age >= 65:
            p_pd += 0.05
        elif age >= 50:
            p_pd += 0.03
        elif age < 40:
            p_pd -= 0.03
    if gender in ("male", "m"):
        p_pd += 0.01
    if isinstance(fam, (list, tuple)) and len(fam) > 0:
        p_pd += 0.05
    # Clamp so demo prior cannot dominate fusion
    p_pd = float(max(0.01, min(0.35, p_pd)))
    return np.asarray([[1.0 - p_pd, p_pd]], dtype=np.float32)


def _risk_from_probs(p: np.ndarray) -> Tuple[str, float]:
    # p shape (1,2) -> PD prob at index 1
    p_pd = float(p[0, 1])
    if p_pd < 0.33:
        rl = "LOW"
    elif p_pd < 0.66:
        rl = "MEDIUM"
    else:
        rl = "HIGH"
    conf = float(max(p[0, 0], p[0, 1]))
    return rl, conf


@infer_bp.post("/api/multimodal/infer")
def multimodal_infer():
    """Fuse speech, spiral, and demographics to produce a PD probability.

    Expects multipart/form-data with optional fields:
      - audio_wav: WAV file blob
      - spiral_image: PNG/JPEG file
      - spiral_image_b64: data URL or raw base64 of image
    """
    try:
        uid, _claims = verify_firebase_id_token(request)
    except AuthError as exc:
        return jsonify({"error": str(exc)}), 401

    _ensure_models_loaded()
    if _FUSION_PREDICTOR is None:
        logger.error("multimodal_infer: models_unavailable")
        return jsonify({"error": "models_unavailable"}), 500

    # Gather inputs
    speech_vec: Optional[np.ndarray] = None
    spiral_img: Optional[np.ndarray] = None
    tmp_path: Optional[str] = None

    try:
        if "audio_wav" in request.files:
            f = request.files["audio_wav"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            speech_vec = _load_wav_to_vec(tmp_path)

        if "spiral_image" in request.files:
            spiral_img = _image_bytes_to_spiral_tensor(request.files["spiral_image"].read())
        elif request.form.get("spiral_image_b64"):
            raw = _dataurl_to_bytes(request.form.get("spiral_image_b64", ""))
            if raw:
                spiral_img = _image_bytes_to_spiral_tensor(raw)

        # Demographics prior from Firestore
        db = firestore.client()
        user_doc = db.collection("users").document(uid).get()
        profile = user_doc.to_dict() if user_doc.exists else {}
        p_demo = _demographics_prior(profile)

        # Predict
        p_s, p_sp, p_d, fused = _FUSION_PREDICTOR.predict(
            speech_vec=speech_vec, spiral_img=spiral_img, p_demo=p_demo
        )

        risk_level, confidence = _risk_from_probs(fused)

        logger.info(
            "multimodal_infer: uid=%s speech_vec=%s spiral_img=%s demo=%s risk=%s confidence=%.3f",
            uid,
            "yes" if speech_vec is not None else "no",
            "yes" if spiral_img is not None else "no",
            "yes" if p_demo is not None else "no",
            risk_level,
            confidence,
        )

        def _pack(p: Optional[np.ndarray]):
            return None if p is None else {"probs": p.tolist(), "pd": float(p[0][1])}

        resp = {
            "probs": fused.tolist(),
            "riskLevel": risk_level,
            "confidence": confidence,
            "speech": _pack(p_s),
            "spiral": _pack(p_sp),
            "demo": _pack(p_d),
            "modelVersion": "mm-fusion-v1",
        }
        return jsonify(resp), 200
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@infer_bp.get("/api/ml/health")
def ml_health():
    """Report ML model readiness and log status.

    Returns JSON with flags for predictor/vectorizer readiness and checkpoint presence.
    """
    runs_dir = os.path.join(os.getcwd(), "runs")
    speech_ckpt_path = os.path.join(runs_dir, "speech.pt")
    spiral_ckpt_path = os.path.join(runs_dir, "spiral.pt")
    spiral_oneclass_ckpt_path = os.path.join(runs_dir, "spiral_oneclass.pt")

    _ensure_models_loaded()

    status = {
        "fusionPredictorReady": _FUSION_PREDICTOR is not None,
        "vectorizerReady": _VECTORIZER is not None,
        "speechCkptFound": os.path.exists(speech_ckpt_path),
        "spiralCkptFound": os.path.exists(spiral_ckpt_path),
        "spiralOneclassCkptFound": os.path.exists(spiral_oneclass_ckpt_path),
    }

    logger.info(
        "ml_health: %s",
        status,
    )
    code = 200 if status["fusionPredictorReady"] else 503
    return jsonify(status), code



    





# -----------------------------

# Multimodal model inference (speech + spiral + demographics)

# -----------------------------



_FUSION_PREDICTOR: Optional[LateFusionPredictor] = None

_VECTORIZER: Optional[UCIPDVectorizer] = None





def _detect_ckpt(path: str) -> Optional[str]:

    return path if os.path.exists(path) else None





def _ensure_models_loaded() -> None:

    global _FUSION_PREDICTOR, _VECTORIZER

    if _FUSION_PREDICTOR is None:

        runs_dir = os.path.join(os.getcwd(), "runs")

        speech_ckpt = _detect_ckpt(os.path.join(runs_dir, "speech.pt"))

        spiral_ckpt = _detect_ckpt(os.path.join(runs_dir, "spiral.pt"))

        spiral_oneclass_ckpt = _detect_ckpt(os.path.join(runs_dir, "spiral_oneclass.pt"))

        logger.info(
            "_ensure_models_loaded: runs_dir=%s speech_ckpt=%s spiral_ckpt=%s spiral_oneclass_ckpt=%s",
            runs_dir,
            bool(speech_ckpt),
            bool(spiral_ckpt),
            bool(spiral_oneclass_ckpt),
        )
        _FUSION_PREDICTOR = LateFusionPredictor(

            speech_ckpt=speech_ckpt,

            spiral_ckpt=spiral_ckpt,

            spiral_oneclass_ckpt=spiral_oneclass_ckpt,

            fusion=WeightedFusion(w_speech=0.55, w_spiral=0.4, w_demo=0.05),

            temperature_speech=1.7,

            temperature_spiral=2.3,

            prob_eps=0.01,

        )

        logger.info("_ensure_models_loaded: LateFusionPredictor initialized")
    if _VECTORIZER is None:

        csv_default = os.path.join("ml_files", "datasets", "voice_recordings", "parkinsons.data")

        if os.path.exists(csv_default):

            _VECTORIZER = UCIPDVectorizer.from_csv(csv_default)

            logger.info("_ensure_models_loaded: UCIPDVectorizer loaded from %s", csv_default)
        else:
            logger.info("_ensure_models_loaded: UCIPDVectorizer CSV not found at %s", csv_default)




def _load_wav_to_vec(tmp_wav_path: str) -> Optional[np.ndarray]:

    if _VECTORIZER is None:

        return None

    try:

        vec = vectorize_wav(tmp_wav_path, _VECTORIZER)

        return vec.astype(np.float32)

    except Exception:

        return None





def _image_bytes_to_spiral_tensor(img_bytes: bytes, size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:

    try:

        img = Image.open(io.BytesIO(img_bytes)).convert("L")

        if img.size != size:

            img = img.resize(size)

        arr = np.asarray(img, dtype=np.float32) / 255.0

        arr = arr.reshape(1, 1, size[1], size[0])

        return arr

    except Exception:

        return None





def _dataurl_to_bytes(data_url: str) -> Optional[bytes]:

    try:

        if data_url.startswith("data:"):

            header, b64 = data_url.split(",", 1)

            return base64.b64decode(b64)

        # treat as base64 plain string

        return base64.b64decode(data_url)

    except Exception:

        return None





def _demographics_prior(user_doc: dict) -> Optional[np.ndarray]:

    # Apply demographics prior only when the user completed the questionnaire

    if not user_doc or not bool(user_doc.get("demographicsCompleted")):

        return None

    age = user_doc.get("age")

    gender = (user_doc.get("gender") or "").lower()

    fam = user_doc.get("familyHistory") or []

    try:

        age = float(age) if age is not None else None

    except Exception:

        age = None

    # Conservative baseline; prevalence is low

    p_pd = 0.10

    if age is not None:

        if age >= 65:

            p_pd += 0.05

        elif age >= 50:

            p_pd += 0.03

        elif age < 40:

            p_pd -= 0.03

    if gender in ("male", "m"):

        p_pd += 0.01

    if isinstance(fam, (list, tuple)) and len(fam) > 0:

        p_pd += 0.05

    # Clamp so demo prior cannot dominate fusion

    p_pd = float(max(0.01, min(0.35, p_pd)))

    return np.asarray([[1.0 - p_pd, p_pd]], dtype=np.float32)





def _risk_from_probs(p: np.ndarray) -> Tuple[str, float]:

    # p shape (1,2) -> PD prob at index 1

    p_pd = float(p[0, 1])

    if p_pd < 0.33:

        rl = "LOW"

    elif p_pd < 0.66:

        rl = "MEDIUM"

    else:

        rl = "HIGH"

    conf = float(max(p[0, 0], p[0, 1]))

    return rl, conf





@infer_bp.post("/api/multimodal/infer")
def multimodal_infer_dup():

    """Fuse speech, spiral, and demographics to produce a PD probability.



    Expects multipart/form-data with optional fields:

      - audio_wav: WAV file blob

      - spiral_image: PNG/JPEG file

      - spiral_image_b64: data URL or raw base64 of image

    """

    try:

        uid, _claims = verify_firebase_id_token(request)

    except AuthError as exc:

        return jsonify({"error": str(exc)}), 401



    _ensure_models_loaded()

    if _FUSION_PREDICTOR is None:

        logger.error("multimodal_infer: models_unavailable")
        return jsonify({"error": "models_unavailable"}), 500



    # Gather inputs

    speech_vec: Optional[np.ndarray] = None

    spiral_img: Optional[np.ndarray] = None

    tmp_path: Optional[str] = None



    try:

        if "audio_wav" in request.files:

            f = request.files["audio_wav"]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

                tmp.write(f.read())

                tmp_path = tmp.name

            speech_vec = _load_wav_to_vec(tmp_path)



        if "spiral_image" in request.files:

            spiral_img = _image_bytes_to_spiral_tensor(request.files["spiral_image"].read())

        elif request.form.get("spiral_image_b64"):

            raw = _dataurl_to_bytes(request.form.get("spiral_image_b64", ""))

            if raw:

                spiral_img = _image_bytes_to_spiral_tensor(raw)



        # Demographics prior from Firestore

        db = firestore.client()

        user_doc = db.collection("users").document(uid).get()

        profile = user_doc.to_dict() if user_doc.exists else {}

        p_demo = _demographics_prior(profile)



        # Predict

        p_s, p_sp, p_d, fused = _FUSION_PREDICTOR.predict(

            speech_vec=speech_vec, spiral_img=spiral_img, p_demo=p_demo

        )



        risk_level, confidence = _risk_from_probs(fused)


        logger.info(
            "multimodal_infer: uid=%s speech_vec=%s spiral_img=%s demo=%s risk=%s confidence=%.3f",
            uid,
            "yes" if speech_vec is not None else "no",
            "yes" if spiral_img is not None else "no",
            "yes" if p_demo is not None else "no",
            risk_level,
            confidence,
        )


        def _pack(p: Optional[np.ndarray]):

            return None if p is None else {"probs": p.tolist(), "pd": float(p[0][1])}



        resp = {

            "probs": fused.tolist(),

            "riskLevel": risk_level,

            "confidence": confidence,

            "speech": _pack(p_s),

            "spiral": _pack(p_sp),

            "demo": _pack(p_d),

            "modelVersion": "mm-fusion-v1",

        }

        return jsonify(resp), 200

    finally:

        if tmp_path and os.path.exists(tmp_path):

            try:

                os.remove(tmp_path)

            except Exception:

                pass




@infer_bp.get("/api/ml/health")
def ml_health_dup():
    """Report ML model readiness and log status.

    Returns JSON with flags for predictor/vectorizer readiness and checkpoint presence.
    """
    runs_dir = os.path.join(os.getcwd(), "runs")
    speech_ckpt_path = os.path.join(runs_dir, "speech.pt")
    spiral_ckpt_path = os.path.join(runs_dir, "spiral.pt")
    spiral_oneclass_ckpt_path = os.path.join(runs_dir, "spiral_oneclass.pt")

    _ensure_models_loaded()

    status = {
        "fusionPredictorReady": _FUSION_PREDICTOR is not None,
        "vectorizerReady": _VECTORIZER is not None,
        "speechCkptFound": os.path.exists(speech_ckpt_path),
        "spiralCkptFound": os.path.exists(spiral_ckpt_path),
        "spiralOneclassCkptFound": os.path.exists(spiral_oneclass_ckpt_path),
    }

    logger.info(
        "ml_health: %s",
        status,
    )
    code = 200 if status["fusionPredictorReady"] else 503
    return jsonify(status), code




