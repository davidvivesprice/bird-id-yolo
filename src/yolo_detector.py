#!/usr/bin/env python3
"""
YOLO-based Real-time Bird Detector

Replaces MOG2 + TFLite classifier with YOLOv8 object detector.
YOLO performs detection + classification in a single pass.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOBirdDetector:
    """YOLO-based bird detector with real-time inference."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics package not available")

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        logger.info(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(str(self.model_path))

        logger.info(f"Model loaded successfully")
        logger.info(f"Model task: {self.model.task}")
        logger.info(f"Model type: {self.model.model.__class__.__name__}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLO detection on frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detections with bbox, class, confidence
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Extract detections from results
        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get class and confidence
                cls_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

                # Get class name
                class_name = result.names[cls_id]

                # Only detect birds - filter out other classes
                if class_name != "bird":
                    continue

                # Convert to detection format
                detection = {
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(x2 - x1),
                    "h": int(y2 - y1),
                    "label": class_name,
                    "confidence": confidence
                }

                detections.append(detection)

        return detections


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    display_frame = frame.copy()

    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        label = det["label"]
        conf = det["confidence"]

        # Draw bounding box
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label background
        label_text = f"{label} ({conf:.1%})"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display_frame, (x, y - 25), (x + label_size[0], y), (0, 255, 0), -1)

        # Draw label text
        cv2.putText(display_frame, label_text, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return display_frame


def write_detections_json(detections: List[Dict], output_path: Path, frame_num: int):
    """Write detections to JSON file (atomic write)."""
    data = {
        "timestamp": time.time(),
        "frame": frame_num,
        "detections": detections
    }

    # Atomic write using temp file + rename
    temp_path = output_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f)
    temp_path.rename(output_path)


def main():
    parser = argparse.ArgumentParser(description="YOLO Bird Detector")
    parser.add_argument("--rtsp", help="RTSP stream URL")
    parser.add_argument("--video", help="Video file path")
    parser.add_argument("--loop", action="store_true", help="Loop video file continuously")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--output", default="/share-yolo/detections.json",
                       help="Output JSON file path")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Minimum confidence threshold")
    parser.add_argument("--resize-width", type=int, default=640,
                       help="Resize frame width for detection (default: 640)")
    parser.add_argument("--resize-height", type=int, default=360,
                       help="Resize frame height for detection (default: 360)")
    parser.add_argument("--roi", type=str, default=None,
                       help="Region of Interest as 'x,y,w,h' (applied before resize)")
    parser.add_argument("--skip-frames", type=int, default=15,
                       help="Process every Nth frame (default: 15 for ~2 FPS on 30fps video)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    args = parser.parse_args()

    # Validate input source
    if not args.rtsp and not args.video:
        logger.error("Either --rtsp or --video must be specified")
        return 1
    if args.rtsp and args.video:
        logger.error("Cannot specify both --rtsp and --video")
        return 1

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Parse ROI if specified
    roi = None
    if args.roi:
        try:
            roi = tuple(map(int, args.roi.split(',')))
            if len(roi) != 4:
                raise ValueError("ROI must have 4 values: x,y,w,h")
        except Exception as e:
            logger.error(f"Invalid ROI format: {e}")
            return 1

    logger.info("=== YOLO Bird Detection System ===")
    if args.rtsp:
        logger.info(f"RTSP Stream: {args.rtsp}")
        video_source = args.rtsp
    else:
        logger.info(f"Video File: {args.video} (loop={args.loop})")
        video_source = args.video
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Confidence Threshold: {args.confidence}")
    logger.info(f"Processing Resolution: {args.resize_width}x{args.resize_height}")
    logger.info(f"Frame Skip: Process every {args.skip_frames} frames")
    if roi:
        logger.info(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    # Initialize detector
    try:
        detector = YOLOBirdDetector(args.model, args.confidence)
    except Exception as e:
        logger.error(f"Failed to initialize YOLO detector: {e}")
        return 1

    # Open video stream
    if args.rtsp:
        logger.info("Opening RTSP stream...")
        cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    else:
        logger.info("Opening video file...")
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        return 1

    logger.info("Video source opened successfully")

    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Stream: {width}x{height} @ {fps:.1f} fps")

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Main detection loop
    frame_count = 0
    last_log_time = time.time()
    detection_count = 0

    logger.info("Starting detection loop...")

    while True:
        ret, frame = cap.read()

        if not ret:
            if args.video and args.loop:
                # Restart video from beginning
                logger.info("Reached end of video, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            elif args.video:
                # End of video, no loop
                logger.info("Reached end of video, stopping...")
                break
            else:
                # RTSP stream failed, reconnect
                logger.warning("Failed to read frame, reconnecting...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
                continue

        frame_count += 1

        # Skip frames for performance
        if frame_count % args.skip_frames != 0:
            continue

        # Store original frame dimensions
        orig_height, orig_width = frame.shape[:2]

        # Apply ROI if specified
        if roi:
            x, y, w, h = roi
            frame_roi = frame[y:y+h, x:x+w]
        else:
            frame_roi = frame
            x, y = 0, 0

        # Resize frame for detection
        frame_resized = cv2.resize(frame_roi, (args.resize_width, args.resize_height))

        # Calculate scale factors for bbox conversion
        scale_x = frame_roi.shape[1] / args.resize_width
        scale_y = frame_roi.shape[0] / args.resize_height

        # Run YOLO detection on resized frame
        detections = detector.detect(frame_resized)

        # Scale detections back to original frame coordinates
        for det in detections:
            det['x'] = int(det['x'] * scale_x) + x
            det['y'] = int(det['y'] * scale_y) + y
            det['w'] = int(det['w'] * scale_x)
            det['h'] = int(det['h'] * scale_y)

        if detections:
            detection_count += len(detections)
            for det in detections:
                logger.info(f"Frame {frame_count}: {det['label']} "
                          f"({det['confidence']:.1%}) at ({det['x']}, {det['y']})")

        # Write detections to JSON
        write_detections_json(detections, output_path, frame_count)

        # Periodic status log
        current_time = time.time()
        if current_time - last_log_time >= 30:
            logger.info(f"Status: {frame_count} frames processed, "
                       f"{detection_count} detections")
            last_log_time = current_time

    cap.release()
    logger.info("Detection stopped")
    return 0


if __name__ == "__main__":
    exit(main())
