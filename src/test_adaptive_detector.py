#!/usr/bin/env python3
"""Test adaptive detector on Downy Woodpecker clip."""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, '/volume1/docker/bird-id/src')

from adaptive_motion_detector import AdaptiveBirdDetector
from motion_detector_mog2 import MotionDetector
from classifier import SpeciesClassifier


def test_detector(video_path: str):
    """Compare old vs adaptive detector."""
    print("=" * 80)
    print("ADAPTIVE MOTION DETECTOR TEST")
    print("=" * 80)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo: {width}x{height} @ {fps} fps")
    print(f"Source: {Path(video_path).name}\n")

    # ROI for this resolution
    roi = [(127, 142), (517, 142), (517, 320), (127, 320)]
    if width != 640:
        scale = width / 640
        roi = [(int(x * scale), int(y * scale)) for x, y in roi]

    # Initialize OLD detector (min_area=500)
    print("Initializing OLD detector (min_area=500)...")
    old_detector = MotionDetector(width, height, mask=roi, min_area=500)

    # Initialize ADAPTIVE detector (min_area=50000, smart filtering)
    print("Initializing ADAPTIVE detector (min_area=50,000, aspect filtering)...")
    adaptive_detector = AdaptiveBirdDetector(
        width, height,
        mask=roi,
        initial_min_area=50000,  # Conservative start based on analysis
        initial_max_area=500000,
        min_aspect=0.3,  # Filter thin streaks
        max_aspect=3.0,
        learning_window=100
    )

    # Initialize classifier
    print("Initializing classifier...\n")
    classifier = SpeciesClassifier(
        Path("/volume1/docker/bird-id/data/models/birds_v1.tflite"),
        Path("/volume1/docker/bird-id/data/models/inat_bird_labels.txt"),
        use_edgetpu=False
    )

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process with OLD detector
    print("=" * 80)
    print("PASS 1: OLD DETECTOR (min_area=500)")
    print("=" * 80)

    frame_count = 0
    old_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 6 == 0:  # Process every 6th frame
            boxes = old_detector.update(frame)

            for box in boxes:
                x, y, w, h = box
                area = w * h
                aspect = w / h if h > 0 else 0
                old_detections.append({
                    'frame': frame_count,
                    'box': box,
                    'area': area,
                    'aspect': aspect
                })

        frame_count += 1

    print(f"Total detections: {len(old_detections)}")
    if old_detections:
        areas = [d['area'] for d in old_detections]
        aspects = [d['aspect'] for d in old_detections]
        print(f"Area range: {min(areas):.0f} - {max(areas):.0f} px")
        print(f"Area mean: {np.mean(areas):.0f} ± {np.std(areas):.0f} px")
        print(f"Aspect range: {min(aspects):.2f} - {max(aspects):.2f}")

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process with ADAPTIVE detector
    print("\n" + "=" * 80)
    print("PASS 2: ADAPTIVE DETECTOR (min_area=50,000 + smart filters)")
    print("=" * 80)

    frame_count = 0
    adaptive_detections = []
    adaptive_classifications = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 6 == 0:  # Process every 6th frame
            boxes = adaptive_detector.update(frame)

            for box in boxes:
                x, y, w, h = box
                area = w * h
                aspect = w / h if h > 0 else 0
                adaptive_detections.append({
                    'frame': frame_count,
                    'box': box,
                    'area': area,
                    'aspect': aspect
                })

                # Classify this detection
                try:
                    # Extract ROI
                    x1 = max(0, x - 5)
                    y1 = max(0, y - 5)
                    x2 = min(width, x + w + 5)
                    y2 = min(height, y + h + 5)

                    roi = frame[y1:y2, x1:x2]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_resized = cv2.resize(roi_rgb, (224, 224))
                    roi_batch = np.expand_dims(roi_resized, axis=0).astype(np.uint8)

                    species, confidence = classifier.classify(roi_batch)

                    adaptive_classifications.append({
                        'frame': frame_count,
                        'box': box,
                        'area': area,
                        'species': species,
                        'confidence': confidence
                    })

                    # Register high-confidence birds for learning
                    if confidence >= 0.7 and species != 'background':
                        adaptive_detector.register_bird_detection(box, species, confidence)
                        print(f"  Frame {frame_count}: {species.split('(')[0].strip():<30} {confidence:>6.1%} "
                              f"(area={area:.0f}px, aspect={aspect:.2f})")

                except Exception as e:
                    print(f"  Classification error: {e}")

        frame_count += 1

    print(f"\nTotal detections: {len(adaptive_detections)}")
    if adaptive_detections:
        areas = [d['area'] for d in adaptive_detections]
        aspects = [d['aspect'] for d in adaptive_detections]
        print(f"Area range: {min(areas):.0f} - {max(areas):.0f} px")
        print(f"Area mean: {np.mean(areas):.0f} ± {np.std(areas):.0f} px")
        print(f"Aspect range: {min(aspects):.2f} - {max(aspects):.2f}")

    # Show learning statistics
    print("\n" + "=" * 80)
    print("ADAPTIVE LEARNING STATISTICS")
    print("=" * 80)

    stats = adaptive_detector.get_statistics()
    if stats['bird_count'] > 0:
        print(f"Learned from: {stats['bird_count']} confirmed bird detections")
        print(f"Bird area: {stats['area_stats']['mean']:.0f} ± {stats['area_stats']['std']:.0f} px")
        print(f"Bird aspect: {stats['aspect_stats']['mean']:.2f} ± {stats['aspect_stats']['std']:.2f}")
        print(f"Current thresholds:")
        print(f"  Area: {stats['current_thresholds']['min_area']:.0f} - {stats['current_thresholds']['max_area']:.0f} px")
        print(f"  Aspect: {stats['current_thresholds']['min_aspect']:.2f} - {stats['current_thresholds']['max_aspect']:.2f}")
        print(f"Species seen: {', '.join(stats['species_seen'])}")
    else:
        print("No high-confidence birds detected for learning")

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\nOLD DETECTOR:")
    print(f"  Total detections: {len(old_detections)}")
    print(f"  (Unable to filter noise - everything passes)")

    print(f"\nADAPTIVE DETECTOR:")
    print(f"  Total detections: {len(adaptive_detections)}")
    print(f"  Noise filtered: {len(old_detections) - len(adaptive_detections)} detections removed")
    print(f"  Reduction: {(1 - len(adaptive_detections)/len(old_detections))*100:.1f}%" if old_detections else "N/A")

    # Show high-confidence classifications
    if adaptive_classifications:
        high_conf = [c for c in adaptive_classifications if c['confidence'] >= 0.7 and 'background' not in c['species'].lower()]
        if high_conf:
            print(f"\nHIGH-CONFIDENCE BIRD DETECTIONS (>=70%):")
            for c in sorted(high_conf, key=lambda x: x['confidence'], reverse=True)[:10]:
                print(f"  Frame {c['frame']:3d}: {c['species'].split('(')[0].strip():<30} {c['confidence']:>6.1%} (area={c['area']:.0f}px)")

    cap.release()
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    video_path = "/volume1/docker/bird-id/data/clips/downy/Birds 11-10-2025, 8.09.23am EST - 11-10-2025, 8.09.28am EST.mp4"
    sys.exit(test_detector(video_path))
