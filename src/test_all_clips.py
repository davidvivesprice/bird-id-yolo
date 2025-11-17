#!/usr/bin/env python3
"""Test classification on all species clips and generate report."""
import sys
from pathlib import Path
from collections import defaultdict
import subprocess


def test_clip(clip_path: Path, species_name: str):
    """Run classification on a single clip and parse results."""
    print(f"\n{'='*70}")
    print(f"Testing: {species_name}")
    print(f"Clip: {clip_path.name}")
    print('='*70)

    output_dir = Path("/volume1/docker/bird-id/data/debug") / f"test_{species_name.replace(' ', '_')}"

    # Run detector with classification
    cmd = [
        "python", "src/motion_detector_mog2.py",
        "--source", str(clip_path),
        "--mask",
        "--frames", "0",
        "--classify",
        "--output", str(output_dir)
    ]

    result = subprocess.run(
        cmd,
        cwd="/volume1/docker/bird-id",
        capture_output=True,
        text=True
    )

    # Parse detections from output
    detections = []
    for line in result.stdout.split('\n'):
        if 'Detected:' in line and 'Frame' in line:
            # Extract species and confidence
            # Format: "Frame 123 - Detected: Species name (confidence: 75.2%)"
            parts = line.split('Detected:')[1].strip()
            species = parts.split('(confidence:')[0].strip()
            confidence = parts.split('(confidence:')[1].split(')')[0].strip()
            detections.append((species, confidence))

    # Get summary stats
    total_detections = len(detections)
    species_counts = defaultdict(int)
    for species, _ in detections:
        species_counts[species] += 1

    print(f"\nResults:")
    print(f"  Total detections: {total_detections}")

    if total_detections == 0:
        print(f"  ✅ No birds detected (expected for baseline)")
    else:
        print(f"\n  Species breakdown:")
        for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections) * 100
            print(f"    {species}: {count} ({percentage:.1f}%)")

        # Show a few example confidences
        print(f"\n  Sample confidences:")
        for species, confidence in detections[:5]:
            print(f"    {species}: {confidence}")

    return {
        'species_name': species_name,
        'clip_path': clip_path.name,
        'total_detections': total_detections,
        'species_counts': dict(species_counts),
        'detections': detections
    }


def main():
    clips_dir = Path("/volume1/docker/bird-id/data/clips")

    # Define test cases
    test_cases = [
        # Known species
        ("Northern Cardinal", clips_dir / "cardinal"),
        ("Chickadee", clips_dir / "chickadees"),
        ("Downy Woodpecker", clips_dir / "downy"),
        ("Tufted Titmouse", clips_dir / "tuftedtitmouse"),

        # Unknown species (to identify)
        ("Unknown Bird 1", clips_dir / "whatisthis?"),
        ("Unknown Bird 2", clips_dir / "brownbird"),

        # Baseline (should be no detections)
        ("Baseline - Feeder Swaying", clips_dir / "baseline"),
    ]

    results = []

    for species_name, folder in test_cases:
        if not folder.exists():
            print(f"\nWARNING: {folder} not found, skipping")
            continue

        # Get first MP4 file
        video_files = list(folder.glob("*.mp4"))
        if not video_files:
            print(f"\nWARNING: No MP4 files in {folder}, skipping")
            continue

        # For baseline, specifically get feeder swaying
        if "Baseline" in species_name:
            swaying_file = folder / "feeder swaying.mp4"
            if swaying_file.exists():
                video_file = swaying_file
            else:
                video_file = video_files[0]
        else:
            video_file = video_files[0]

        result = test_clip(video_file, species_name)
        results.append(result)

    # Print summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)

    for result in results:
        print(f"\n{result['species_name']}:")
        print(f"  Clip: {result['clip_path']}")
        print(f"  Detections: {result['total_detections']}")

        if result['total_detections'] > 0:
            top_species = sorted(result['species_counts'].items(), key=lambda x: x[1], reverse=True)[0]
            print(f"  Top classification: {top_species[0]} ({top_species[1]} detections)")

    print("\n" + "="*70)
    print("Test complete! Check debug folders for frame outputs.")
    print("="*70)


if __name__ == "__main__":
    sys.exit(main())
