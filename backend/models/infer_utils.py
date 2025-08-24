from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch

from .fusion_model import MultimodalPDModel


def export_onnx(model: MultimodalPDModel, num_demo: int, num_speech: int, onnx_path: str) -> None:
    model.eval()
    demo = torch.randn(1, num_demo)
    speech = torch.randn(1, num_speech)
    spiral = torch.randn(1, 1, 224, 224)
    torch.onnx.export(
        model,
        (demo, speech, spiral),
        onnx_path,
        input_names=["demo", "speech", "spiral"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "demo": {0: "batch"},
            "speech": {0: "batch"},
            "spiral": {0: "batch"},
            "logits": {0: "batch"},
        },
    )


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


class OnnxMultimodalInfer:
    def __init__(self, onnx_path: str):
        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

    def predict(
        self,
        demo: Optional[np.ndarray],
        speech: Optional[np.ndarray],
        spiral_img: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        inputs = {}
        if demo is None:
            demo = np.zeros((1, self.sess.get_inputs()[0].shape[1]), dtype=np.float32)
        if speech is None:
            speech = np.zeros((1, self.sess.get_inputs()[1].shape[1]), dtype=np.float32)
        if spiral_img is None:
            spiral_img = np.zeros((1, 1, 224, 224), dtype=np.float32)
        inputs["demo"] = demo.astype(np.float32)
        inputs["speech"] = speech.astype(np.float32)
        inputs["spiral"] = spiral_img.astype(np.float32)
        logits = self.sess.run(["logits"], inputs)[0]
        probs = softmax(logits, axis=-1)
        return logits, probs


