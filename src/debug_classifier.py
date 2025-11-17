#!/usr/bin/env python3
"""Debug classifier by visualizing detections and crops."""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from classifier import SpeciesClassifier
from motion_detector_mog2 import MotionDetector


def extract_and_prepare_roi(frame: np.ndarray, box: tuple, padding: int = 5):
    """Extract ROI from frame and prepare for classification."""
    x, y, w, h = box
    height, width = frame.shape[:2]

    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    # Extract ROI
    roi = frame[y1:y2, x1:x2]

    # Convert BGR to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Resize to 224x224
    roi_resized = cv2.resize(roi_rgb, (224, 224))

    # Add batch dimension
    roi_batch = np.expand_dims(roi_resized, axis=0)

    return roi_batch.astype(np.uint8), roi, (x1, y1, x2, y2)


def main():
    parser = argparse.ArgumentParser(description="Debug classifier on video")
    parser.add_argument("--source", type=str, required=True, help="Video file path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for debug images")
    parser.add_argument("--model", type=Path,
                       default=Path("/volume1/docker/bird-id/data/models/birds_v1.tflite"))
    parser.add_argument("--labels", type=Path,
                       default=Path("/volume1/docker/bird-id/data/models/inat_bird_labels.txt"))
    parser.add_argument("--padding", type=int, default=5, help="ROI padding")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize classifier
    print("Initializing classifier...")
    classifier = SpeciesClassifier(args.model, args.labels, use_edgetpu=False)

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.source}")
        sys.exit(1)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {width}x{height} @ {fps} fps")

    # Initialize motion detector
    roi = [(127, 142), (517, 142), (517, 320), (127, 320)]  # Default ROI for substream
    # Scale ROI to video resolution if needed
    if width != 640:
        scale = width / 640
        roi = [(int(x * scale), int(y * scale)) for x, y in roi]

    detector = MotionDetector(width, height, mask=roi, min_area=500)

    frame_count = 0
    detection_count = 0
    all_results = []

    print(f"\nProcessing video with padding={args.padding}px...")
    print("=" * 80)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

        # Detect motion (every 6th frame)
        if frame_count % 6 == 0:
            boxes = detector.update(frame)

            if boxes:
                print(f"\nFrame {frame_count}: Detected {len(boxes)} bird(s)")

                for i, box in enumerate(boxes):
                    x, y, w, h = box

                    # Try different padding values
                    for pad in [0, 5, 10, 20]:
                        roi_batch, roi_original, (x1, y1, x2, y2) = extract_and_prepare_roi(frame, box, padding=pad)
                        species, confidence = classifier.classify(roi_batch)

                        result = {
                            'frame': frame_count,
                            'detection': i,
                            'padding': pad,
                            'bbox': box,
                            'species': species,
                            'confidence': confidence
                        }
                        all_results.append(result)

                        print(f"  Detection {i} (pad={pad}px): {species.split('(')[0].strip():<30} {confidence:>6.1%}")

                    # Save visualization for padding=5 (default)
                    roi_batch, roi_original, (x1, y1, x2, y2) = extract_and_prepare_roi(frame, box, padding=args.padding)
                    species, confidence = classifier.classify(roi_batch)

                    # Draw on frame
                    frame_viz = frame.copy()
                    cv2.rectangle(frame_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    label = f"{species.split('(')[0].strip()}: {confidence:.1%}"
                    cv2.putText(frame_viz, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Save full frame with bbox
                    frame_out = args.output / f"frame_{frame_count:04d}_det_{i}_full.jpg"
                    cv2.imwrite(str(frame_out), frame_viz)

                    # Save original ROI (before resize)
                    roi_out = args.output / f"frame_{frame_count:04d}_det_{i}_roi_original.jpg"
                    cv2.imwrite(str(roi_out), cv2.cvtColor(roi_original, cv2.COLOR_RGB2BGR))

                    # Save 224x224 crop (what classifier sees)
                    crop_out = args.output / f"frame_{frame_count:04d}_det_{i}_crop_224.jpg"
                    cv2.imwrite(str(crop_out), cv2.cvtColor(roi_batch[0], cv2.COLOR_RGB2BGR))

                    print(f"    Saved: {frame_out.name}, {roi_out.name}, {crop_out.name}")

                    detection_count += 1

        frame_count += 1

    cap.release()

    print("\n" + "=" * 80)
    print(f"Processed {frame_count} frames, {detection_count} detections")
    print(f"Debug images saved to: {args.output}")

    # Print summary of padding comparison
    if all_results:
        print("\n" + "=" * 80)
        print("PADDING COMPARISON:")
        print("=" * 80)

        from collections import defaultdict
        by_padding = defaultdict(list)
        for r in all_results:
            by_padding[r['padding']].append(r['confidence'])

        for pad in sorted(by_padding.keys()):
            confidences = by_padding[pad]
            avg_conf = np.mean(confidences)
            max_conf = np.max(confidences)
            print(f"Padding {pad:2d}px: avg={avg_conf:.1%}, max={max_conf:.1%}, "
                  f"samples={len(confidences)}")

        # Find best detections
        print("\n" + "=" * 80)
        print("TOP 5 DETECTIONS:")
        print("=" * 80)
        sorted_results = sorted(all_results, key=lambda x: x['confidence'], reverse=True)[:5]
        for r in sorted_results:
            print(f"Frame {r['frame']:4d}, Det {r['detection']}, Pad {r['padding']:2d}px: "
                  f"{r['species'].split('(')[0].strip():<30} {r['confidence']:>6.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
