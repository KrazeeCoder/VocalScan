"""Lightweight VocalScan backend for demo purposes."""

import os
import json
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})


class SimpleVocalScan:
    """Simplified VocalScan for demo without ML dependencies."""
    
    def predict(self, audio_bytes, sample_type="voice", duration=10):
        """Simulate prediction without ML libraries."""
        
        # Simple scoring based on sample type and duration
        if sample_type in ["cough", "breath"]:
            respiratory_score = min(0.8, 0.1 + (len(audio_bytes) / 100000) * 0.3)
            neurological_score = 0.0
        elif sample_type in ["voice", "sustained", "sentence"]:
            respiratory_score = 0.0
            neurological_score = min(0.8, 0.1 + (duration / 20) * 0.3)
        else:
            respiratory_score = 0.2
            neurological_score = 0.15
        
        scores = {
            "respiratory": round(respiratory_score, 3),
            "neurological": round(neurological_score, 3)
        }
        
        max_score = max(scores.values())
        if max_score < 0.33:
            risk_level = "low"
        elif max_score < 0.66:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        confidence = min(0.95, 0.6 + (duration / 30) * 0.3)
        
        interpretation = self._generate_interpretation(scores, risk_level, sample_type)
        
        return {
            "scores": scores,
            "confidence": round(confidence, 3),
            "riskLevel": risk_level,
            "modelVersion": "vocalscan-lite-v1.0",
            "sampleType": sample_type,
            "interpretation": interpretation,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_interpretation(self, scores, risk_level, sample_type):
        """Generate interpretation."""
        interpretation = {
            "summary": "",
            "details": [],
            "nextSteps": [],
            "disclaimer": "This is a pattern analysis tool, not a medical diagnosis. Consult healthcare professionals for medical advice."
        }
        
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
                "Consider consultation with healthcare provider if patterns persist"
            ]
        else:
            interpretation["summary"] = "Notable patterns detected that may warrant attention."
            interpretation["nextSteps"] = [
                "Consider consultation with a healthcare provider",
                "Document any symptoms or voice changes"
            ]
        
        respiratory_score = scores.get("respiratory", 0)
        neurological_score = scores.get("neurological", 0)
        
        if respiratory_score > 0.3:
            interpretation["details"].append(
                f"Respiratory analysis shows patterns of interest (score: {respiratory_score:.2f})"
            )
        
        if neurological_score > 0.3:
            interpretation["details"].append(
                f"Voice analysis shows patterns of interest (score: {neurological_score:.2f})"
            )
        
        if not interpretation["details"]:
            interpretation["details"].append("Analysis shows patterns within normal ranges.")
        
        return interpretation


# Initialize model
model = SimpleVocalScan()


@app.route('/health')
def health():
    return jsonify({"status": "ok", "message": "VocalScan Lite Backend Running"})


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in form-data"}), 400

    file_storage = request.files["file"]
    audio_bytes = file_storage.read()

    sample_rate_str = request.form.get("sampleRate", "16000")
    duration_sec_str = request.form.get("durationSec", "10")
    sample_type = request.form.get("sampleType", "voice")

    try:
        duration_sec = float(duration_sec_str)
    except ValueError:
        duration_sec = 10

    result = model.predict(audio_bytes, sample_type, duration_sec)
    
    # Add record ID
    record_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result["recordId"] = record_id
    
    return jsonify(result)


@app.route('/infer', methods=['POST'])
def infer():
    """Legacy endpoint for compatibility."""
    return predict()


@app.route('/delete/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    """Delete record endpoint."""
    return jsonify({"message": f"Record {record_id} deleted successfully"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🎤 VocalScan Lite Backend starting on port {port}")
    print("✅ Ready to accept audio analysis requests")
    app.run(host='0.0.0.0', port=port, debug=True)
