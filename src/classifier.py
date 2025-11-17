#!/usr/bin/env python3
"""Placeholder species classifier wrapper (TFLite-ready)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except Exception:  # fallback if package missing
    tflite = None  # type: ignore

DEFAULT_LABELS = {0: "unknown"}


class SpeciesClassifier:
    """Thin wrapper around a TFLite model (EdgeTPU-ready)."""

    def __init__(self, model_path: Path, labels_path: Optional[Path] = None, use_edgetpu: bool = False):
        self.model_path = model_path
        self.labels = self._load_labels(labels_path)
        self.interpreter = self._load_interpreter(model_path, use_edgetpu)
        if self.interpreter:
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.input_details = []
            self.output_details = []

    def _load_labels(self, labels_path: Optional[Path]):
        if labels_path and labels_path.exists():
            with labels_path.open() as f:
                if labels_path.suffix == ".json":
                    mapping = json.load(f)
                    return {int(k): v for k, v in mapping.items()}
                return {idx: line.strip() for idx, line in enumerate(f) if line.strip()}
        return DEFAULT_LABELS

    def _load_interpreter(self, model_path: Path, use_edgetpu: bool):
        if tflite is None:
            return None
        delegates = []
        edgetpu_available = False
        if use_edgetpu:
            try:
                delegates.append(tflite.load_delegate('libedgetpu.so.1'))
                edgetpu_available = True
            except Exception:
                print("Warning: EdgeTPU delegate not available; falling back to CPU")
                # Switch to non-EdgeTPU model if using EdgeTPU model
                if '_edgetpu' in str(model_path):
                    model_path = Path(str(model_path).replace('_edgetpu.tflite', '.tflite'))
                    print(f"Switching to CPU model: {model_path}")
        interpreter = tflite.Interpreter(model_path=str(model_path), experimental_delegates=delegates or None)
        interpreter.allocate_tensors()
        return interpreter

    def classify(self, image) -> Tuple[str, float]:
        if not self.interpreter:
            return "unavailable", 0.0
        input_info = self.input_details[0]
        tensor = image
        if tensor.dtype != input_info["dtype"]:
            tensor = tensor.astype(input_info["dtype"])
        self.interpreter.set_tensor(input_info["index"], tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # Normalize if output is uint8 (0-255 range)
        if output.dtype == np.uint8:
            output = output.astype(np.float32) / 255.0

        top_idx = int(output.argmax())
        score = float(output[top_idx])
        return self.labels.get(top_idx, "unknown"), score


def _cli():
    parser = argparse.ArgumentParser(description="Run a single inference on an image.")
    parser.add_argument("model", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--edgetpu", action="store_true")
    args = parser.parse_args()

    import cv2
    classifier = SpeciesClassifier(args.model, args.labels, args.edgetpu)
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit("Failed to load image")
    label, score = classifier.classify(img)
    print(f"{label}: {score:.2f}")


if __name__ == "__main__":
    _cli()
