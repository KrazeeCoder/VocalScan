from __future__ import annotations

import os
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import joblib
import pandas as pd


def _extract_features_for_wav(wav_path: str) -> Dict[str, float]:
    from backend.models.voice_features import extract_ucipd_from_wav
    try:
        feats = extract_ucipd_from_wav(wav_path)
    except Exception as e:
        raise RuntimeError(f"feature extraction failed for {wav_path}: {e}")

    # Add simple RMS using librosa (optional)
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        rms = librosa.feature.rms(y=y)[0]
        feats["rms_db_mean"] = float(20*np.log10(np.mean(rms)+1e-6))
        feats["rms_db_max"] = float(20*np.log10(np.max(rms)+1e-6))
    except Exception:
        pass
    return feats


def _ensure_wav(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".wav":
        return str(p)
    # Convert to wav with ffmpeg
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-i", str(p), "-ar", "16000", "-ac", "1", tmp_wav], check=True)
    return tmp_wav


def _scan_dataset(root: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for label_name, label_val in (("HC_AH", 0), ("PD_AH", 1)):
        d = Path(root) / label_name
        if not d.exists():
            continue
        for fn in d.rglob("*.wav"):
            items.append((str(fn), label_val))
        # Allow non-wav with conversion
        for fn in d.rglob("*.m4a"):
            items.append((str(fn), label_val))
        for fn in d.rglob("*.mp3"):
            items.append((str(fn), label_val))
    if not items:
        raise SystemExit(f"No audio files found under {root}/HC_AH or {root}/PD_AH")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF on WAV features (HC/PD)")
    parser.add_argument("--root", type=str, default=os.path.join("ml_files", "datasets", "parkinsonA"), help="Root with HC_AH and PD_AH")
    parser.add_argument("--out", type=str, default=os.path.join("runs", "uci_rf.joblib"), help="Output bundle path")
    args = parser.parse_args()

    items = _scan_dataset(args.root)

    # Extract features
    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    for path, y in items:
        wav = _ensure_wav(path)
        feats = _extract_features_for_wav(wav)
        rows.append(feats)
        labels.append(int(y))

    # Align to shared columns
    df = pd.DataFrame(rows)
    cols = sorted(df.columns)
    X = df[cols].fillna(0.0).astype(float).values
    y = np.asarray(labels, dtype=int)

    # Train classifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=400, random_state=42)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"accuracy: {acc:.4f} (n={len(y)})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "cols": cols}, args.out)
    print(f"saved bundle to {args.out}")


if __name__ == "__main__":
    main()


