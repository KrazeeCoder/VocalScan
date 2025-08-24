## VocalScan

VocalScan is a Flask-based web application for collecting data relevant to Parkinson's disease assessment. It lets authenticated users:

- Record and upload voice samples
- Draw and upload spiral drawings
- Enter demographics
- View recent results on a dashboard

The app stores uploads in Firebase Cloud Storage and metadata in Firestore under a simple, user-centric schema. Model scoring is currently a placeholder to enable end-to-end flows without ML.

## Tech stack

- Backend: Flask, firebase-admin (Python), CORS
- Frontend: Vanilla JS, Firebase Web SDK (compat), Tailwind via CDN
- Data: Firestore (Native mode), Firebase Storage

## Repository layout

- `backend/app/`
  - `__init__.py` application factory, CORS, Firebase Admin init
  - `pages.py` server-rendered routes (`/login`, `/record`, `/spiral`, `/dashboard`, `/profile`)
  - `infer.py` placeholder analysis endpoints (`/infer`, `/spiral/infer`) and Firestore writes
  - `firebase_admin_init.py` service account discovery/initialization
- `backend/templates/` HTML templates
- `backend/static/js/` frontend scripts
- `start-backend.bat` convenience script for Windows dev

## Prerequisites

- Python 3.11+
- A Firebase project with:
  - Authentication (Email/Password + Google enabled as desired)
  - Firestore (Native mode) enabled
  - Cloud Storage enabled
- A service account key (JSON) with Firebase Admin permissions

## Configuration

1) Service account

- Place your service account JSON at the repo root, or set one of:
  - `GOOGLE_APPLICATION_CREDENTIALS` (absolute path to the JSON)
  - `FIREBASE_SERVICE_ACCOUNT_FILE` (absolute path to the JSON)
- The app auto-detects a JSON that matches `vocalscan-firebase-adminsdk-*.json` in repo root if present.

2) Frontend Firebase config

- Edit `backend/static/js/firebase-init.js` and ensure the config matches your project (apiKey, authDomain, projectId, appId, etc.).
- For Storage, set `storageBucket` to your project’s default bucket. For newer projects this may be `your-project.firebasestorage.app`.

Example (replace with your values):

```javascript
const firebaseConfig = {
  apiKey: "...",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project",
  storageBucket: "your-project.firebasestorage.app",
  appId: "..."
};
```

## Running locally

Windows (recommended for this repo):

```bash
start-backend.bat
```

Manual steps:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
set FLASK_APP=backend.app.main
python -m flask run --host=0.0.0.0 --port=8080
```

App runs at `http://127.0.0.1:8080`.

## Security rules

Storage (development example allowing authenticated users). Adjust for production as needed:

```js
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /{allPaths=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

Firestore (allow users to read/write their own profile and read their own submissions):

```js
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
      match /voiceRecordings/{recordId} {
        allow read: if request.auth != null && request.auth.uid == userId;
      }
      match /spiralDrawings/{drawingId} {
        allow read: if request.auth != null && request.auth.uid == userId;
      }
    }
  }
}
```

Note: The backend uses Firebase Admin SDK for writes and is not restricted by Firestore rules; these rules govern client reads/writes (e.g., demographics).

## Data model

Firestore documents:

```
users/{uid}
  createdAt (server timestamp)
  updatedAt (server timestamp)
  lastRecordingAt (server timestamp)
  lastSpiralAt (server timestamp)
  firstName, lastName, age, sex, notes
  voiceRecordingPaths: array<string>           # Storage paths: audio/{uid}/...
  spiralDrawingPaths: array<string>            # Storage paths: spirals/{uid}/...
  voiceRecordings/{recordId}
    createdAt (server timestamp)
    storagePath (audio/{uid}/{recordId}.webm)
    durationSec, sampleRate
    scores { respiratory, neurological }
    confidence, riskLevel, modelVersion, status, timestamp
  spiralDrawings/{drawingId}
    createdAt (server timestamp)
    storagePath (spirals/{uid}/{drawingId}.png)
    scores { respiratory, neurological }
    confidence, riskLevel, modelVersion, status, timestamp
```

Storage paths:

- Audio: `audio/{uid}/rec_<timestamp>.webm`
- Spirals: `spirals/{uid}/spiral_<timestamp>.png`

## Pages

- `/login` Email/Password + Google sign-in
- `/record` Record voice, upload to Storage, call `/infer`
- `/spiral` Canvas spiral tool, upload PNG to Storage, call `/spiral/infer`
- `/dashboard` Combined history view (voice + spirals) from Firestore
- `/profile` Demographics form stored to `users/{uid}`

## API endpoints

- `POST /infer`
  - Auth: Firebase ID token (`Authorization: Bearer <token>`)
  - Body: multipart/form-data: `file` (audio), `sampleRate`, `durationSec`, `recordId` (optional)
  - Response: `{ recordId, scores, confidence, riskLevel, storagePath, modelVersion }`

- `POST /spiral/infer`
  - Auth: Firebase ID token
  - Body: multipart/form-data: `file` (png), `drawingId` (optional)
  - Response: `{ drawingId, scores, confidence, riskLevel, storagePath, modelVersion }`

Both endpoints compute deterministic placeholder scores and persist metadata under `users/{uid}`.

## Troubleshooting

- Uploads stuck at 0%:
  - Ensure you are signed in (uploads require `request.auth != null` with the sample rules).
  - Confirm `storageBucket` in `firebase-init.js` matches your project bucket (e.g., `your-project.firebasestorage.app` for newer projects).
  - Check browser console for lines like “Audio upload error” or “Spiral upload error”.
  - Verify Storage rules and network connectivity.

- Firestore reads show empty dashboard:
  - Confirm Firestore rules allow the signed-in user to read `users/{uid}` and its subcollections.

## Deployment

- Production server example (Linux):

```bash
pip install -r backend/requirements.txt
gunicorn -w 2 -b 0.0.0.0:8080 backend.wsgi:app
```

- Set `ALLOWED_ORIGINS` and `LOG_LEVEL` as needed (see `backend/app/__init__.py`).

## Disclaimer

This application uses placeholder analysis and is provided for demonstration and data collection. It is not a medical device and does not provide medical advice.


