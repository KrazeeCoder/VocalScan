"""Audio feature extraction for VocalScan."""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract features for respiratory and neurological analysis."""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        
    def preprocess_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio from bytes."""
        try:
            # Try to load with soundfile first
            import io
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # Resample to target sample rate
            if sr != self.target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.target_sr)
                
            # Normalize
            audio_data = librosa.util.normalize(audio_data)
            
            return audio_data, self.target_sr
            
        except Exception as e:
            # Fallback: assume raw audio data
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            if len(audio_data) == 0:
                raise ValueError("Could not parse audio data")
            return librosa.util.normalize(audio_data), self.target_sr
    
    def extract_respiratory_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract features for respiratory anomaly detection."""
        features = {}
        
        # Log-mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr, 
            n_mels=80,
            hop_length=int(sr * 0.01),  # 10ms hop
            win_length=int(sr * 0.025)  # 25ms window
        )
        log_mel = librosa.power_to_db(mel_spec)
        
        # Statistical features from mel spectrogram
        features['mel_mean'] = np.mean(log_mel)
        features['mel_std'] = np.std(log_mel)
        features['mel_max'] = np.max(log_mel)
        features['mel_min'] = np.min(log_mel)
        features['mel_range'] = features['mel_max'] - features['mel_min']
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate (indicates breathy/rough texture)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Energy features
        rms = librosa.feature.rms(y=audio_data)[0]
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_max'] = np.max(rms)
        
        # Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        return features
    
    def extract_neurological_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract features for neurological voice analysis (Parkinson's-style)."""
        features = {}
        
        # MFCC features (most important for voice analysis)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, threshold=0.1)
        
        # Extract fundamental frequency
        f0_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                f0_values.append(pitch)
        
        if f0_values:
            f0_array = np.array(f0_values)
            features['f0_mean'] = np.mean(f0_array)
            features['f0_std'] = np.std(f0_array)
            features['f0_min'] = np.min(f0_array)
            features['f0_max'] = np.max(f0_array)
            features['f0_range'] = features['f0_max'] - features['f0_min']
            
            # Jitter (pitch variability)
            if len(f0_array) > 1:
                jitter = np.mean(np.abs(np.diff(f0_array)) / features['f0_mean'])
                features['jitter'] = jitter
            else:
                features['jitter'] = 0.0
        else:
            # No pitch detected
            features.update({
                'f0_mean': 0.0, 'f0_std': 0.0, 'f0_min': 0.0, 
                'f0_max': 0.0, 'f0_range': 0.0, 'jitter': 0.0
            })
        
        # Shimmer approximation (amplitude variability)
        rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
        if len(rms) > 1:
            shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms) if np.mean(rms) > 0 else 0.0
            features['shimmer'] = shimmer
        else:
            features['shimmer'] = 0.0
        
        # Harmonic-to-noise ratio approximation
        # Using spectral features as proxy
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
        
        if len(spectral_bandwidth) > 0 and len(spectral_centroids) > 0:
            hnr_proxy = np.mean(spectral_centroids) / (np.mean(spectral_bandwidth) + 1e-8)
            features['hnr_proxy'] = hnr_proxy
        else:
            features['hnr_proxy'] = 0.0
        
        # Voice quality indicators
        features['voice_energy_mean'] = np.mean(rms)
        features['voice_energy_std'] = np.std(rms)
        
        # Pause analysis (silence detection)
        frame_length = 2048
        hop_length = 512
        silence_threshold = 0.01
        
        frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames**2, axis=0)
        silent_frames = frame_energies < silence_threshold
        
        if len(silent_frames) > 0:
            features['silence_ratio'] = np.sum(silent_frames) / len(silent_frames)
        else:
            features['silence_ratio'] = 0.0
        
        return features
    
    def extract_all_features(self, audio_bytes: bytes, sample_type: str = "voice") -> Dict[str, float]:
        """Extract all relevant features based on sample type."""
        audio_data, sr = self.preprocess_audio(audio_bytes)
        
        if sample_type in ["cough", "breath"]:
            return self.extract_respiratory_features(audio_data, sr)
        else:  # voice, sustained vowel, sentence
            return self.extract_neurological_features(audio_data, sr)
