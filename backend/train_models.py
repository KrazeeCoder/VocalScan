"""Data preparation and model training for VocalScan."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
import joblib
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from app.models.audio_features import AudioFeatureExtractor


class VocalScanTrainer:
    """Train models for VocalScan using available datasets."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "app/models/trained_models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.feature_extractor = AudioFeatureExtractor()
        
        os.makedirs(self.models_dir, exist_ok=True)
    
    def prepare_respiratory_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare respiratory data from available sources."""
        print("Preparing respiratory data...")
        
        # For demo: create synthetic data that mimics real respiratory patterns
        # In production, this would load from PhysioNet/Coswara datasets
        
        n_normal = 800
        n_abnormal = 200
        
        # Normal respiratory patterns (lower variability, regular patterns)
        normal_features = []
        for _ in range(n_normal):
            features = {
                'mel_mean': np.random.normal(-30, 5),
                'mel_std': np.random.normal(8, 2),
                'mel_range': np.random.normal(40, 8),
                'spectral_centroid_mean': np.random.normal(2000, 300),
                'spectral_rolloff_mean': np.random.normal(4000, 500),
                'zcr_mean': np.random.normal(0.1, 0.02),
                'energy_mean': np.random.normal(0.05, 0.01),
                'chroma_mean': np.random.normal(0.3, 0.05)
            }
            # Add more features to reach ~20 total
            for i in range(12):
                features[f'extra_{i}'] = np.random.normal(0, 1)
            
            normal_features.append(list(features.values()))
        
        # Abnormal respiratory patterns (higher variability, irregular)
        abnormal_features = []
        for _ in range(n_abnormal):
            features = {
                'mel_mean': np.random.normal(-25, 8),  # Higher energy
                'mel_std': np.random.normal(12, 4),    # More variability
                'mel_range': np.random.normal(55, 15), # Wider range
                'spectral_centroid_mean': np.random.normal(2500, 600),  # Different spectrum
                'spectral_rolloff_mean': np.random.normal(5000, 1000),
                'zcr_mean': np.random.normal(0.15, 0.05),  # More irregular
                'energy_mean': np.random.normal(0.08, 0.03),  # More variable energy
                'chroma_mean': np.random.normal(0.25, 0.1)    # Different harmonic content
            }
            # Add more features
            for i in range(12):
                features[f'extra_{i}'] = np.random.normal(0, 2)  # Higher variance
            
            abnormal_features.append(list(features.values()))
        
        # Combine data
        X = np.array(normal_features + abnormal_features)
        y = np.array([0] * n_normal + [1] * n_abnormal)
        
        return X, y
    
    def prepare_neurological_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare neurological voice data."""
        print("Preparing neurological voice data...")
        
        # For demo: create synthetic data mimicking Parkinson's voice patterns
        # In production, this would load from UCI Parkinson's dataset
        
        n_healthy = 600
        n_parkinson = 400
        
        # Healthy voice patterns
        healthy_features = []
        for _ in range(n_healthy):
            features = {}
            
            # MFCC features (20 coefficients, mean and std)
            for i in range(20):
                features[f'mfcc_{i}_mean'] = np.random.normal(0, 1)
                features[f'mfcc_{i}_std'] = np.random.normal(1, 0.3)
            
            # Voice quality features
            features.update({
                'f0_mean': np.random.normal(150, 30),      # Normal pitch
                'f0_std': np.random.normal(10, 3),         # Low variability
                'jitter': np.random.normal(0.005, 0.002),  # Low jitter
                'shimmer': np.random.normal(0.03, 0.01),   # Low shimmer
                'hnr_proxy': np.random.normal(15, 3),      # Good HNR
                'voice_energy_mean': np.random.normal(0.1, 0.02),
                'silence_ratio': np.random.normal(0.1, 0.05)
            })
            
            healthy_features.append(list(features.values()))
        
        # Parkinson's voice patterns (characteristic changes)
        parkinson_features = []
        for _ in range(n_parkinson):
            features = {}
            
            # MFCC features (different distribution)
            for i in range(20):
                features[f'mfcc_{i}_mean'] = np.random.normal(0, 1.5)  # More variable
                features[f'mfcc_{i}_std'] = np.random.normal(1.5, 0.5)
            
            # Parkinson's-specific changes
            features.update({
                'f0_mean': np.random.normal(140, 40),      # More variable pitch
                'f0_std': np.random.normal(20, 8),         # Higher variability
                'jitter': np.random.normal(0.015, 0.008),  # Higher jitter
                'shimmer': np.random.normal(0.08, 0.03),   # Higher shimmer
                'hnr_proxy': np.random.normal(10, 4),      # Lower HNR
                'voice_energy_mean': np.random.normal(0.07, 0.03),  # More variable
                'silence_ratio': np.random.normal(0.15, 0.08)  # More pauses
            })
            
            parkinson_features.append(list(features.values()))
        
        # Combine data
        X = np.array(healthy_features + parkinson_features)
        y = np.array([0] * n_healthy + [1] * n_parkinson)
        
        return X, y
    
    def train_respiratory_model(self) -> Dict[str, Any]:
        """Train the respiratory anomaly detection model."""
        print("Training respiratory model...")
        
        X, y = self.prepare_respiratory_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try different models
        models = {
            'logistic': LogisticRegression(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            
            print(f"{name}: AUC = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        print(f"Best respiratory model: {best_name} (AUC: {best_score:.3f})")
        
        # Save the best model
        joblib.dump(best_model, os.path.join(self.models_dir, 'respiratory_model.pkl'))
        joblib.dump(scaler, os.path.join(self.models_dir, 'respiratory_scaler.pkl'))
        
        return {"model_type": best_name, "auc_score": best_score}
    
    def train_neurological_model(self) -> Dict[str, Any]:
        """Train the neurological voice analysis model."""
        print("Training neurological model...")
        
        X, y = self.prepare_neurological_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try different models
        models = {
            'logistic': LogisticRegression(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            
            print(f"{name}: AUC = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        print(f"Best neurological model: {best_name} (AUC: {best_score:.3f})")
        
        # Save the best model
        joblib.dump(best_model, os.path.join(self.models_dir, 'neurological_model.pkl'))
        joblib.dump(scaler, os.path.join(self.models_dir, 'neurological_scaler.pkl'))
        
        return {"model_type": best_name, "auc_score": best_score}
    
    def train_anomaly_detector(self) -> Dict[str, Any]:
        """Train unsupervised anomaly detector as fallback."""
        print("Training anomaly detector...")
        
        # Combine some features from both datasets for general anomaly detection
        X_resp, _ = self.prepare_respiratory_data()
        X_neuro, _ = self.prepare_neurological_data()
        
        # Use subset of features
        X_combined = np.column_stack([
            X_resp[:, :15],  # First 15 respiratory features
            X_neuro[:, :15]  # First 15 neurological features
        ])
        
        # Train isolation forest
        anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        anomaly_detector.fit(X_combined)
        
        # Save model
        joblib.dump(anomaly_detector, os.path.join(self.models_dir, 'anomaly_detector.pkl'))
        
        return {"model_type": "isolation_forest", "contamination": 0.1}
    
    def train_all_models(self):
        """Train all models and save results."""
        print("=== VocalScan Model Training ===")
        
        results = {}
        
        # Train respiratory model
        results['respiratory'] = self.train_respiratory_model()
        
        # Train neurological model
        results['neurological'] = self.train_neurological_model()
        
        # Train anomaly detector
        results['anomaly'] = self.train_anomaly_detector()
        
        print("\n=== Training Complete ===")
        for model_type, result in results.items():
            print(f"{model_type}: {result}")
        
        return results


if __name__ == "__main__":
    # Run training
    trainer = VocalScanTrainer()
    trainer.train_all_models()
