"""Machine learning models for VocalScan."""

import numpy as np
import pickle
import os
from typing import Dict, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import joblib
from .audio_features import AudioFeatureExtractor


class VocalScanModel:
    """Main model class for VocalScan predictions."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'trained_models')
        self.feature_extractor = AudioFeatureExtractor()
        
        # Model components
        self.respiratory_model = None
        self.neurological_model = None
        self.respiratory_scaler = None
        self.neurological_scaler = None
        self.anomaly_detector = None
        
        # Initialize with placeholder models if trained ones don't exist
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models (load if available, create placeholders otherwise)."""
        try:
            self._load_models()
        except (FileNotFoundError, Exception):
            # Create placeholder models for demo
            self._create_placeholder_models()
    
    def _load_models(self):
        """Load pre-trained models from disk."""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError("Model directory not found")
        
        # Load respiratory model
        resp_model_path = os.path.join(self.model_dir, 'respiratory_model.pkl')
        resp_scaler_path = os.path.join(self.model_dir, 'respiratory_scaler.pkl')
        
        if os.path.exists(resp_model_path) and os.path.exists(resp_scaler_path):
            self.respiratory_model = joblib.load(resp_model_path)
            self.respiratory_scaler = joblib.load(resp_scaler_path)
        
        # Load neurological model
        neuro_model_path = os.path.join(self.model_dir, 'neurological_model.pkl')
        neuro_scaler_path = os.path.join(self.model_dir, 'neurological_scaler.pkl')
        
        if os.path.exists(neuro_model_path) and os.path.exists(neuro_scaler_path):
            self.neurological_model = joblib.load(neuro_model_path)
            self.neurological_scaler = joblib.load(neuro_scaler_path)
        
        # Load anomaly detector
        anomaly_path = os.path.join(self.model_dir, 'anomaly_detector.pkl')
        if os.path.exists(anomaly_path):
            self.anomaly_detector = joblib.load(anomaly_path)
    
    def _create_placeholder_models(self):
        """Create placeholder models for demo purposes."""
        
        # Create dummy training data
        n_samples = 1000
        n_features_resp = 20
        n_features_neuro = 50
        
        # Respiratory model (binary classification)
        X_resp = np.random.randn(n_samples, n_features_resp)
        y_resp = np.random.randint(0, 2, n_samples)
        
        self.respiratory_scaler = StandardScaler()
        X_resp_scaled = self.respiratory_scaler.fit_transform(X_resp)
        
        self.respiratory_model = LogisticRegression(random_state=42)
        self.respiratory_model.fit(X_resp_scaled, y_resp)
        
        # Neurological model (binary classification)
        X_neuro = np.random.randn(n_samples, n_features_neuro)
        y_neuro = np.random.randint(0, 2, n_samples)
        
        self.neurological_scaler = StandardScaler()
        X_neuro_scaled = self.neurological_scaler.fit_transform(X_neuro)
        
        self.neurological_model = LogisticRegression(random_state=42)
        self.neurological_model.fit(X_neuro_scaled, y_neuro)
        
        # Anomaly detector (unsupervised)
        X_combined = np.random.randn(n_samples, 30)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector.fit(X_combined)
        
        # Save placeholder models
        self._save_models()
    
    def _save_models(self):
        """Save models to disk."""
        os.makedirs(self.model_dir, exist_ok=True)
        
        if self.respiratory_model:
            joblib.dump(self.respiratory_model, os.path.join(self.model_dir, 'respiratory_model.pkl'))
            joblib.dump(self.respiratory_scaler, os.path.join(self.model_dir, 'respiratory_scaler.pkl'))
        
        if self.neurological_model:
            joblib.dump(self.neurological_model, os.path.join(self.model_dir, 'neurological_model.pkl'))
            joblib.dump(self.neurological_scaler, os.path.join(self.model_dir, 'neurological_scaler.pkl'))
        
        if self.anomaly_detector:
            joblib.dump(self.anomaly_detector, os.path.join(self.model_dir, 'anomaly_detector.pkl'))
    
    def _extract_and_align_features(self, features: Dict[str, float], feature_names: list) -> np.ndarray:
        """Extract features in the correct order for model prediction."""
        feature_vector = []
        for name in feature_names:
            feature_vector.append(features.get(name, 0.0))
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_respiratory(self, audio_bytes: bytes) -> Tuple[float, float]:
        """Predict respiratory anomaly risk."""
        if not self.respiratory_model:
            # Fallback to anomaly detection
            return self._predict_anomaly_fallback(audio_bytes, "respiratory")
        
        # Extract features
        features = self.feature_extractor.extract_respiratory_features(
            *self.feature_extractor.preprocess_audio(audio_bytes)
        )
        
        # Get expected feature names from model training
        expected_features = getattr(self.respiratory_model, 'feature_names_', list(features.keys())[:20])
        feature_vector = self._extract_and_align_features(features, expected_features)
        
        # Scale features
        feature_vector_scaled = self.respiratory_scaler.transform(feature_vector)
        
        # Predict
        risk_proba = self.respiratory_model.predict_proba(feature_vector_scaled)[0]
        risk_score = risk_proba[1] if len(risk_proba) > 1 else risk_proba[0]
        confidence = max(risk_proba) * 0.8 + 0.2  # Add some uncertainty
        
        return float(risk_score), float(confidence)
    
    def predict_neurological(self, audio_bytes: bytes) -> Tuple[float, float]:
        """Predict neurological voice anomaly risk."""
        if not self.neurological_model:
            # Fallback to anomaly detection
            return self._predict_anomaly_fallback(audio_bytes, "neurological")
        
        # Extract features
        features = self.feature_extractor.extract_neurological_features(
            *self.feature_extractor.preprocess_audio(audio_bytes)
        )
        
        # Get expected feature names from model training
        expected_features = getattr(self.neurological_model, 'feature_names_', list(features.keys())[:50])
        feature_vector = self._extract_and_align_features(features, expected_features)
        
        # Scale features
        feature_vector_scaled = self.neurological_scaler.transform(feature_vector)
        
        # Predict
        risk_proba = self.neurological_model.predict_proba(feature_vector_scaled)[0]
        risk_score = risk_proba[1] if len(risk_proba) > 1 else risk_proba[0]
        confidence = max(risk_proba) * 0.8 + 0.2  # Add some uncertainty
        
        return float(risk_score), float(confidence)
    
    def _predict_anomaly_fallback(self, audio_bytes: bytes, analysis_type: str) -> Tuple[float, float]:
        """Fallback prediction using anomaly detection."""
        try:
            if analysis_type == "respiratory":
                features = self.feature_extractor.extract_respiratory_features(
                    *self.feature_extractor.preprocess_audio(audio_bytes)
                )
            else:
                features = self.feature_extractor.extract_neurological_features(
                    *self.feature_extractor.preprocess_audio(audio_bytes)
                )
            
            # Use first 30 features for anomaly detection
            feature_vector = np.array(list(features.values())[:30]).reshape(1, -1)
            
            if self.anomaly_detector:
                anomaly_score = self.anomaly_detector.decision_function(feature_vector)[0]
                # Convert to probability-like score (0-1)
                risk_score = max(0, min(1, (anomaly_score + 0.5) / 1.0))
                confidence = 0.6  # Lower confidence for anomaly detection
            else:
                # Ultimate fallback - use feature statistics
                feature_mean = np.mean(list(features.values()))
                feature_std = np.std(list(features.values()))
                risk_score = min(1.0, max(0.0, (abs(feature_mean) + feature_std) / 10.0))
                confidence = 0.4
            
            return float(risk_score), float(confidence)
            
        except Exception:
            # Final fallback
            return 0.3, 0.3
    
    def predict(self, audio_bytes: bytes, sample_type: str) -> Dict[str, Any]:
        """Main prediction method."""
        
        # Determine analysis type based on sample type
        if sample_type in ["cough", "breath"]:
            resp_score, resp_conf = self.predict_respiratory(audio_bytes)
            neuro_score, neuro_conf = 0.0, 0.0
        elif sample_type in ["voice", "sustained", "sentence"]:
            neuro_score, neuro_conf = self.predict_neurological(audio_bytes)
            resp_score, resp_conf = 0.0, 0.0
        else:
            # Run both analyses
            resp_score, resp_conf = self.predict_respiratory(audio_bytes)
            neuro_score, neuro_conf = self.predict_neurological(audio_bytes)
        
        # Calculate overall metrics
        scores = {
            "respiratory": round(resp_score, 3),
            "neurological": round(neuro_score, 3)
        }
        
        # Overall confidence is weighted average
        weights = {"respiratory": resp_conf, "neurological": neuro_conf}
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            confidence = sum(score * weights[key] for key, score in scores.items()) / total_weight
        else:
            confidence = 0.5
        
        # Risk level based on highest score
        max_score = max(scores.values())
        if max_score < 0.33:
            risk_level = "low"
        elif max_score < 0.66:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "scores": scores,
            "confidence": round(confidence, 3),
            "risk_level": risk_level,
            "model_version": "vocalscan-v1.0"
        }


# Global model instance
_model_instance = None

def get_model() -> VocalScanModel:
    """Get singleton model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = VocalScanModel()
    return _model_instance
