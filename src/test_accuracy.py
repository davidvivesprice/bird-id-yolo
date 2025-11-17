#!/usr/bin/env python3
"""Test classification accuracy on known species samples."""
import sys
from pathlib import Path
import cv2
import numpy as np
from classifier import SpeciesClassifier


def test_species_folder(classifier, species_folder: Path, expected_species: str):
    """Test all samples in a species folder."""
    print(f"\n=== Testing {species_folder.name} ===")
    print(f"Expected: {expected_species}")

    samples = sorted(species_folder.glob("*.jpg"))
    if not samples:
        print("  No samples found!")
        return 0, 0

    correct = 0
    results = []

    for sample_path in samples:
        # Load image (already 224x224 RGB)
        img_bgr = cv2.imread(str(sample_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Add batch dimension
        img_batch = np.expand_dims(img_rgb, axis=0).astype(np.uint8)

        # Classify
        species, confidence = classifier.classify(img_batch)

        # Check if correct
        is_correct = expected_species.lower() in species.lower()
        if is_correct:
            correct += 1

        results.append((sample_path.name, species, confidence, is_correct))

    # Print results
    for filename, species, confidence, is_correct in results:
        status = "✅" if is_correct else "❌"
        print(f"  {status} {filename}: {species.split('(')[0].strip()} ({confidence:.1%})")

    accuracy = (correct / len(samples)) * 100 if samples else 0
    print(f"\nAccuracy: {correct}/{len(samples)} ({accuracy:.1f}%)")

    return correct, len(samples)


def main():
    model_path = Path("/volume1/docker/bird-id/data/models/birds_v1.tflite")
    labels_path = Path("/volume1/docker/bird-id/data/models/inat_bird_labels.txt")
    samples_dir = Path("/volume1/docker/bird-id/data/test_samples")

    print("=== Bird Classification Accuracy Test ===")
    print(f"Model: {model_path.name}")
    print(f"Samples: {samples_dir}")

    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = SpeciesClassifier(model_path, labels_path, use_edgetpu=False)
    print("Classifier ready!")

    # Species mappings (folder name → expected label substring)
    species_map = {
        "Northern_Cardinal": "Cardinal",
        "Chickadee": "Chickadee",
        "Downy_Woodpecker": "Downy",
        "Tufted_Titmouse": "Titmouse",
    }

    total_correct = 0
    total_samples = 0

    for folder_name, expected in species_map.items():
        species_folder = samples_dir / folder_name
        if not species_folder.exists():
            print(f"\nWARNING: {folder_name} folder not found")
            continue

        correct, count = test_species_folder(classifier, species_folder, expected)
        total_correct += correct
        total_samples += count

    # Overall stats
    print("\n" + "="*50)
    print("=== OVERALL RESULTS ===")
    print(f"Total Correct: {total_correct}/{total_samples}")
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print("="*50)


if __name__ == "__main__":
    sys.exit(main())
