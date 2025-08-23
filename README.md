# VocalScan - AI-Powered Voice Health Analysis

![VocalScan](https://img.shields.io/badge/VocalScan-v1.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9+-green.svg)
![React](https://img.shields.io/badge/React-18+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

VocalScan is an AI-powered web application that analyzes voice patterns to identify potential health markers. It records 10-20 second voice samples and provides pattern-match risk scores for respiratory anomalies and neuro-voice dysphonia signals.

## 🎯 Features

- **Real-time Voice Recording**: Record voice samples directly in the browser
- **Multiple Sample Types**: 
  - Sustained "aaah" sounds
  - Read sentences
  - Cough/breath audio
  - General voice analysis
- **AI-Powered Analysis**: 
  - Respiratory anomaly detection
  - Neurological voice pattern analysis
  - Confidence scoring
- **User-Friendly Interface**: Clean, modern React frontend with real-time waveform visualization
- **Privacy-First**: No permanent storage of audio data
- **Pattern Detection**: Returns "low/medium/high" risk levels with explanations

## 🏗️ Architecture

### Frontend (React/Next.js)
- Web Audio API for recording
- Real-time waveform visualization
- Responsive design with Tailwind CSS
- TypeScript for type safety

### Backend (Python/Flask)
- FastAPI-style Flask application
- Machine learning pipeline with:
  - Audio feature extraction (librosa)
  - Multiple ML models (scikit-learn, LightGBM)
  - Anomaly detection fallback

### ML Pipeline
- **Respiratory Analysis**: Log-mel spectrograms, spectral features
- **Neurological Analysis**: MFCCs, pitch analysis, jitter/shimmer
- **Models**: Logistic Regression, SVM, Random Forest, LightGBM
- **Fallback**: Isolation Forest for unsupervised anomaly detection

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### Automated Setup
```bash
python setup.py
```

### Manual Setup

#### Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python train_models.py  # Train initial models
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run build
```

### Running the Application

#### Option 1: Using batch files (Windows)
```bash
# Start backend
start-backend.bat

# Start frontend (new terminal)
start-frontend.bat
```

#### Option 2: Manual start
```bash
# Terminal 1 - Backend
cd backend
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
python app/main.py

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

## 📊 API Endpoints

### POST /predict
Main prediction endpoint for voice analysis.

**Request:**
```
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: audio file (WebM, WAV, MP3)
sampleRate: 16000
durationSec: recording duration
sampleType: voice|sustained|sentence|cough|breath
```

**Response:**
```json
{
  "recordId": "rec_2025-08-23_10-30-45",
  "modelVersion": "vocalscan-v1.0",
  "sampleType": "voice",
  "scores": {
    "respiratory": 0.234,
    "neurological": 0.156
  },
  "confidence": 0.823,
  "riskLevel": "low",
  "interpretation": {
    "summary": "Low likelihood of concerning patterns detected.",
    "details": ["Analysis shows patterns within normal ranges."],
    "nextSteps": ["Continue regular health monitoring"],
    "disclaimer": "This is a pattern analysis tool, not a medical diagnosis."
  },
  "timestamp": "2025-08-23T10:30:45.123Z"
}
```

### DELETE /delete/<record_id>
Delete user's record for privacy.

## 🧠 Machine Learning Details

### Feature Extraction

#### Respiratory Features
- Log-mel spectrogram (80 mel bins, 25ms window, 10ms hop)
- Spectral centroid and rolloff
- Zero crossing rate
- Energy statistics
- Chroma features

#### Neurological Features
- 20 MFCC coefficients (mean and std)
- Fundamental frequency (F0) analysis
- Jitter and shimmer approximation
- Harmonic-to-noise ratio proxy
- Voice quality indicators
- Pause/silence analysis

### Model Training
The system trains multiple models and selects the best performing:
- Logistic Regression
- Support Vector Machine
- Random Forest
- LightGBM

Fallback: Isolation Forest for unsupervised anomaly detection.

### Datasets (for production)
- **Respiratory**: PhysioNet Respiratory Sound Database, Coswara
- **Neurological**: UCI Parkinson's voice datasets, Common Voice (healthy samples)

## 🔧 Configuration

### Environment Variables

#### Backend (.env)
```
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000
PORT=8080
```

#### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8080
```

## 📁 Project Structure

```
VocalScan/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   │   ├── audio_features.py    # Feature extraction
│   │   │   ├── ml_models.py        # ML models
│   │   │   └── trained_models/     # Saved models
│   │   ├── main.py                 # Flask app
│   │   ├── infer.py               # Prediction endpoints
│   │   └── auth_mock.py           # Development auth
│   ├── requirements.txt
│   └── train_models.py            # Model training script
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   ├── WaveformVisualizer.tsx
│   │   │   ├── ResultsDisplay.tsx
│   │   │   └── SampleTypeSelector.tsx
│   │   ├── page.tsx               # Main app component
│   │   └── layout.tsx
│   ├── package.json
│   └── tailwind.config.js
├── setup.py                       # Automated setup
├── start-backend.bat             # Windows start script
├── start-frontend.bat            # Windows start script
└── README.md
```

## 🔒 Security & Privacy

- No permanent audio storage
- Client-side audio processing where possible
- Mock authentication for development
- CORS protection
- Input validation and sanitization

## ⚠️ Disclaimer

**Important**: VocalScan is a pattern analysis tool for research and screening purposes only. It does not provide medical diagnoses and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Troubleshooting

### Common Issues

1. **Microphone not working**: Check browser permissions for microphone access
2. **Module not found errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`, `npm install`)
3. **CORS errors**: Check that ALLOWED_ORIGINS includes your frontend URL
4. **Model training fails**: Models will fall back to demo versions automatically

### Development Tips

- Use `npm run dev` for frontend development with hot reload
- Backend runs on port 8080, frontend on port 3000
- Check browser console for client-side errors
- Check terminal output for server-side errors

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the API documentation

---

Built with ❤️ for health technology innovation