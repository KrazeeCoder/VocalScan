# Multimodal PD Model (Demographics + Voice + Spiral)

This folder contains a lightweight, production-friendly multimodal architecture for Parkinson's detection compatible with the datasets in ml_files/datasets/.

## Components
- encoders.py: modality encoders
  - DemographicsMLP: tabular demographic/clinical features
  - SpeechMLP: precomputed voice feature vectors (e.g., UCI PD)
  - SpiralCNN: small CNN over spiral images rendered from coordinate .txt files
  - FusionHead: gating-based fusion that tolerates missing modalities
- fusion_model.py: MultimodalPDModel wiring encoders + fusion head
- datasets.py:
  - SpeechFeatureDataset: loads voice_recordings/parkinsons.data-style CSV
  - SpiralDrawingDataset: renders spiral_drawings/**.txt into grayscale images
- train.py: starters for unimodal training over speech, spiral (.txt), or PNG spiral (one-class)
- infer_utils.py: ONNX export and inference helper
- spiral_oneclass.py: Deep SVDD-style one-class model for PNG spirals (PD-only)

## Quickstart (Speech-only training)
```
# Train on provided UCI PD CSV
python -m backend.models.train --speech_csv ml_files/datasets/voice_recordings/parkinsons.data --epochs 15 --save_ckpt runs/speech.pt

# Train on spiral drawings (place .txt files under a root)
python -m backend.models.train --spiral_root ml_files/datasets/spiral_drawings/ --epochs 15 --save_ckpt runs/spiral.pt
# Train one-class spiral model on PNGs (PD-only images)
python -m backend.models.train --spiral_img_root ml_files/datasets/spiral_data/ --epochs 40 --save_ckpt runs/spiral_oneclass.pt

# Zero-config: run without flags to auto-detect datasets and train what's available
python -m backend.models.train
```

## ONNX Export
```
from backend.models.fusion_model import MultimodalPDModel
from backend.models.infer_utils import export_onnx

num_demo = 4
num_speech = 22  # set based on your CSV feature count
model = MultimodalPDModel(num_demo, num_speech)
export_onnx(model, num_demo, num_speech, "multimodal.onnx")
```

## Notes
- Spiral .txt files are rendered to 224x224 grayscale images; adjust size/line width in SpiralConfig.
- Fusion supports missing modalities by gating; you can feed zeros for absent inputs at inference.
- For production, export to ONNX and run with onnxruntime CPU provider.

## Voice feature extraction from WAVs
Requirements: `praat-parselmouth` for jitter/shimmer/HNR. Install:
```bash
pip install praat-parselmouth onnxruntime numpy
```

Example to extract UCI-PD features from a WAV, align to your CSV header, standardize, and run ONNX inference:
```python
from backend.models.voice_features import UCIPDVectorizer, extract_ucipd_from_wav
from backend.models.infer_utils import OnnxMultimodalInfer

# Build vectorizer stats and column order from your training CSV
vec = UCIPDVectorizer.from_csv("ml_files/datasets/voice_recordings/parkinsons.data")

# Extract per-WAV UCI-style features
feat_dict = extract_ucipd_from_wav("path/to/voice.wav")  # returns dict of columns
speech_x = vec.transform(feat_dict)  # shape (1, D) float32, standardized

# Inference (voice-only)
onnx = OnnxMultimodalInfer("multimodal.onnx")
logits, probs = onnx.predict(demo=None, speech=speech_x, spiral_img=None)
print(probs)
```

Notes:
- The vectorizer must be built from the same CSV used for training to match column order and normalization.
- Nonlinear measures (RPDE, DFA, D2, PPE, spread1/2) are approximations suitable for inference consistency.

## Export ONNX after training
```python
from backend.models.fusion_model import MultimodalPDModel
from backend.models.infer_utils import export_onnx
import torch

ckpt = torch.load("runs/speech.pt", map_location="cpu")
model = MultimodalPDModel(num_demo_features=ckpt["num_demo"], num_speech_features=ckpt["num_speech"]) 
model.load_state_dict(ckpt["model_state"])
export_onnx(model, ckpt["num_demo"], ckpt["num_speech"], "multimodal.onnx")
```

## Late fusion with unpaired datasets (speech + spiral)
You can train speech-only and spiral-only models separately, then fuse their probabilities:
```python
import numpy as np
from backend.models.late_fusion import LateFusionPredictor, WeightedFusion
from backend.models.voice_features import UCIPDVectorizer, vectorize_wav

# Load models
pred = LateFusionPredictor(
    speech_ckpt="runs/speech.pt",      # from speech-only training
    spiral_ckpt="runs/spiral.pt",      # optional: from .txt spiral training
    spiral_oneclass_ckpt="runs/spiral_oneclass.pt",  # optional: from PNG one-class training
)

# Prepare inputs
vec = UCIPDVectorizer.from_csv("ml_files/datasets/voice_recordings/parkinsons.data")
speech_x = vectorize_wav("path/to/voice.wav", vec)  # (1, D)
# Spiral input: either (1,1,H,W) tensor for CNN model, or (1,1,H,W)/(1,H,W) for one-class predictor
spiral_img = np.zeros((1, 1, 224, 224), dtype=np.float32)

# Predict and fuse
p_speech, p_spiral, p_fused = pred.predict(speech_vec=speech_x, spiral_img=spiral_img)
print("speech:", p_speech, "spiral:", p_spiral, "fused:", p_fused)
```
- Default fusion is equal-weight probability averaging. Adjust with `WeightedFusion(w_speech=0.6, w_spiral=0.4)` when constructing `LateFusionPredictor` if one modality is more reliable.

## Notes on Spiral One-Class (PNG) Model
- Trains only on PD-positive spirals from `ml_files/datasets/spiral_data/` (Dynamic/Static tests).
- Objective pulls embeddings toward a learned center; probability maps from distance via a sigmoid.
- Use alongside voice and demographics via late fusion for robust predictions.
