#!/usr/bin/env python3
"""Test bird classifier on sample images."""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from classifier import SpeciesClassifier


def preprocess_image(image_path: Path, target_size=(224, 224)):
    """Load and preprocess image for bird classification."""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(img, target_size)

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    # Model expects uint8
    return img.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Test bird classifier")
    parser.add_argument("image", type=Path, help="Path to bird image")
    parser.add_argument("--model", type=Path,
                       default=Path("/volume1/docker/bird-id/data/models/birds_v1.tflite"),
                       help="Path to TFLite model")
    parser.add_argument("--labels", type=Path,
                       default=Path("/volume1/docker/bird-id/data/models/inat_bird_labels.txt"),
                       help="Path to labels file")
    parser.add_argument("--edgetpu", action="store_true", help="Use EdgeTPU")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Loading labels: {args.labels}")

    # Initialize classifier
    classifier = SpeciesClassifier(args.model, args.labels, use_edgetpu=args.edgetpu)

    # Preprocess image
    print(f"\nLoading image: {args.image}")
    img_tensor = preprocess_image(args.image)
    print(f"Image shape: {img_tensor.shape}")
    print(f"Image dtype: {img_tensor.dtype}")

    # Classify
    print("\nClassifying...")
    species, confidence = classifier.classify(img_tensor)

    # Display result
    print(f"\n=== Result ===")
    print(f"Species: {species}")
    print(f"Confidence: {confidence:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
