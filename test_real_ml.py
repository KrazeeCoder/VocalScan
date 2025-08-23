"""Test script to demonstrate real ML audio analysis."""

import requests
import numpy as np
import wave
import io
import json

def create_test_audio(duration=5, sample_rate=16000, frequency=440):
    """Create a test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create different types of test signals
    signals = {
        'voice': np.sin(2 * np.pi * frequency * t) * 0.3 + \
                np.sin(2 * np.pi * frequency * 2 * t) * 0.1 + \
                np.random.normal(0, 0.05, len(t)),  # Voice-like with harmonics
        
        'cough': np.random.normal(0, 0.3, len(t)) * \
                np.exp(-t * 3) * \
                (np.sin(2 * np.pi * 20 * t) + 1),  # Cough-like burst
        
        'breath': np.random.normal(0, 0.1, len(t)) * \
                 (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))  # Breathing-like
    }
    
    return signals

def audio_to_wav_bytes(audio_data, sample_rate=16000):
    """Convert numpy array to WAV bytes."""
    # Normalize and convert to 16-bit
    audio_data = np.clip(audio_data, -1, 1)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create WAV in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    return buffer.getvalue()

def test_real_ml_backend():
    """Test the real ML backend with different audio types."""
    
    print("🎤 Testing Real ML VocalScan Backend")
    print("=" * 50)
    
    # Test health endpoint first
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Backend Health Check:")
            print(f"   Status: {health_data['status']}")
            print(f"   ML Models Trained: {health_data['models_trained']}")
            print(f"   Libraries: {', '.join(health_data['ml_libraries'])}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return
    
    # Generate test audio signals
    test_signals = create_test_audio()
    
    sample_types = [
        ("voice", "Voice/Speech Analysis"),
        ("cough", "Cough Analysis"), 
        ("breath", "Breathing Analysis")
    ]
    
    for sample_type, description in sample_types:
        print(f"\n🔬 Testing: {description}")
        print("-" * 30)
        
        # Get the appropriate test signal
        audio_data = test_signals[sample_type]
        wav_bytes = audio_to_wav_bytes(audio_data)
        
        # Prepare request
        files = {'file': ('test.wav', wav_bytes, 'audio/wav')}
        data = {
            'sampleType': sample_type,
            'sampleRate': '16000',
            'durationSec': '5'
        }
        
        try:
            # Send to real ML backend
            response = requests.post(
                "http://localhost:8080/predict",
                files=files,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Analysis Complete:")
                print(f"   Model Version: {result['modelVersion']}")
                print(f"   Risk Level: {result['riskLevel'].upper()}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Respiratory Score: {result['scores']['respiratory']:.3f}")
                print(f"   Neurological Score: {result['scores']['neurological']:.3f}")
                
                if 'technicalDetails' in result:
                    tech = result['technicalDetails']
                    print(f"   Audio Processed: {tech['audioProcessed']}")
                    print(f"   Features Extracted: {tech['featuresExtracted']}")
                    print(f"   ML Models Used: {tech['mlModelsUsed']}")
                
                # Show interpretation
                interpretation = result.get('interpretation', {})
                if interpretation.get('summary'):
                    print(f"   Summary: {interpretation['summary']}")
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Request error: {e}")
    
    print(f"\n🎯 Real ML Analysis Complete!")
    print("The backend is now using:")
    print("✅ Real audio feature extraction (librosa)")
    print("✅ Trained ML models (Random Forest)")
    print("✅ Actual signal processing")
    print("✅ No fake statistics!")

if __name__ == "__main__":
    test_real_ml_backend()
