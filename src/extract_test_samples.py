#!/usr/bin/env python3
"""Extract 224x224 bird samples from clips for classification testing."""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from motion_detector_mog2 import MotionDetector, extract_and_prepare_roi, load_config


def extract_samples(clip_path: Path, output_dir: Path, species_name: str, max_samples: int = 10):
    """Extract bird samples from a clip."""
    print(f"\n=== Processing {species_name} ===")
    print(f"Clip: {clip_path.name}")

    # Load config for ROI
    config_path = Path("/volume1/docker/bird-id/config.yaml")
    config = load_config(config_path)
    motion_config = config.get("motion", {})

    # Get ROI coordinates for main stream (1920x1080)
    roi_coords = motion_config.get("roi_main", [(380, 425), (1550, 425), (1550, 960), (380, 960)])

    # Open video
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {clip_path}")
        return 0

    ret, frame = cap.read()
    if not ret:
        print(f"ERROR: Cannot read frame from {clip_path}")
        return 0

    height, width = frame.shape[:2]
    print(f"Resolution: {width}x{height}")

    # Initialize motion detector
    detector = MotionDetector(
        width, height,
        mask=roi_coords,
        method='mog2',
        history=500,
        min_area=500,
        mog2_var_threshold=16,
        detect_shadows=True
    )

    samples_saved = 0
    frame_id = 0

    # Create species output directory
    species_dir = output_dir / species_name
    species_dir.mkdir(parents=True, exist_ok=True)

    while samples_saved < max_samples:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect motion
        boxes = detector.update(frame)

        # Save first bird from each frame with detection
        if boxes:
            for box in boxes[:1]:  # Just take first detection per frame
                # Extract and resize ROI to 224x224
                roi_tensor = extract_and_prepare_roi(frame, box, target_size=(224, 224))
                roi_image = roi_tensor[0]  # Remove batch dimension

                # Convert RGB back to BGR for saving
                roi_bgr = cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR)

                # Save sample
                sample_path = species_dir / f"{species_name}_{samples_saved:03d}_frame{frame_id:04d}.jpg"
                cv2.imwrite(str(sample_path), roi_bgr)
                print(f"  Saved: {sample_path.name}")

                samples_saved += 1
                if samples_saved >= max_samples:
                    break

        frame_id += 1

    cap.release()
    print(f"Extracted {samples_saved} samples for {species_name}")
    return samples_saved


def main():
    parser = argparse.ArgumentParser(description="Extract 224x224 test samples from bird clips")
    parser.add_argument("--output", type=Path, default=Path("/volume1/docker/bird-id/data/test_samples"),
                       help="Output directory for test samples")
    parser.add_argument("--samples-per-species", type=int, default=10,
                       help="Number of samples to extract per species")
    args = parser.parse_args()

    # Known species clips
    clips_dir = Path("/volume1/docker/bird-id/data/clips")
    species_clips = {
        "Northern_Cardinal": clips_dir / "cardinal",
        "Chickadee": clips_dir / "chickadees",
        "Downy_Woodpecker": clips_dir / "downy",
        "Tufted_Titmouse": clips_dir / "tuftedtitmouse",
    }

    args.output.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    for species_name, species_dir in species_clips.items():
        if not species_dir.exists():
            print(f"WARNING: {species_dir} not found, skipping")
            continue

        # Get first video file
        video_files = list(species_dir.glob("*.mp4"))
        if not video_files:
            print(f"WARNING: No MP4 files in {species_dir}, skipping")
            continue

        clip_path = video_files[0]
        count = extract_samples(clip_path, args.output, species_name, args.samples_per_species)
        total_samples += count

    print(f"\n=== Extraction Complete ===")
    print(f"Total samples: {total_samples}")
    print(f"Output directory: {args.output}")
    print(f"\nReady for classification testing!")


if __name__ == "__main__":
    sys.exit(main())
