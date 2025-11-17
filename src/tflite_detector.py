#!/usr/bin/env python3
"""
TFLite Edge TPU Bird Detector
Uses Coral Edge TPU for real-time object detection
"""

import argparse
import time
import logging
import json
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np

# Try to import Edge TPU libraries
try:
    from pycoral.adapters import common, detect
    from pycoral.utils.edgetpu import make_interpreter
    TPU_AVAILABLE = True
except ImportError:
    import tflite_runtime.interpreter as tflite
    TPU_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TFLiteDetector:
    """TFLite detector with optional Edge TPU acceleration"""

    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.5, use_tpu: bool = True):
        self.confidence_threshold = confidence_threshold
        self.use_tpu = use_tpu and TPU_AVAILABLE

        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Initialize interpreter
        if self.use_tpu:
            logger.info("Initializing Edge TPU interpreter...")
            self.interpreter = make_interpreter(model_path)
        else:
            logger.info("Initializing CPU TFLite interpreter...")
            self.interpreter = tflite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

        logger.info(f"Model input size: {self.input_width}x{self.input_height}")
        logger.info(f"Using {'Edge TPU' if self.use_tpu else 'CPU'}")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run detection on frame"""
        orig_height, orig_width = frame.shape[:2]

        # Resize frame to model input size
        frame_resized = cv2.resize(frame, (self.input_width, self.input_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Set input tensor
        if self.use_tpu:
            common.set_input(self.interpreter, frame_rgb)
        else:
            input_data = np.expand_dims(frame_rgb, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get detections
        if self.use_tpu:
            objs = detect.get_objects(self.interpreter, self.confidence_threshold)
        else:
            # Parse output tensors manually for CPU
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

            objs = []
            for i in range(len(scores)):
                if scores[i] >= self.confidence_threshold:
                    objs.append({
                        'bbox': boxes[i],
                        'class_id': int(classes[i]),
                        'score': float(scores[i])
                    })

        # Convert to our detection format
        detections = []
        for obj in objs:
            if self.use_tpu:
                bbox = obj.bbox
                class_id = obj.id
                score = obj.score
                ymin, xmin, ymax, xmax = bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax
            else:
                bbox = obj['bbox']
                class_id = obj['class_id']
                score = obj['score']
                ymin, xmin, ymax, xmax = bbox

            # Scale to original image coordinates
            x = int(xmin * orig_width)
            y = int(ymin * orig_height)
            w = int((xmax - xmin) * orig_width)
            h = int((ymax - ymin) * orig_height)

            label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"

            # Only keep bird detections
            if label.lower() == "bird":
                detections.append({
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "label": label,
                    "confidence": score
                })

        return detections


def write_detections_json(detections: List[Dict], output_path: Path, frame_count: int):
    """Write detections to JSON file atomically"""
    data = {
        "timestamp": time.time(),
        "frame": frame_count,
        "detections": detections
    }

    # Write atomically
    temp_path = output_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f)
    temp_path.rename(output_path)


def main():
    parser = argparse.ArgumentParser(description="TFLite Edge TPU Bird Detector")
    parser.add_argument("--rtsp", help="RTSP stream URL")
    parser.add_argument("--video", help="Video file path")
    parser.add_argument("--loop", action="store_true", help="Loop video file continuously")
    parser.add_argument("--model", required=True, help="Path to TFLite model (.tflite)")
    parser.add_argument("--labels", required=True, help="Path to labels file (.txt)")
    parser.add_argument("--output", default="/share-yolo/detections.json",
                       help="Output JSON file path")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Minimum confidence threshold")
    parser.add_argument("--no-tpu", action="store_true",
                       help="Disable Edge TPU, use CPU only")
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

    logger.info("=== TFLite Edge TPU Bird Detection System ===")
    if args.rtsp:
        logger.info(f"RTSP Stream: {args.rtsp}")
        video_source = args.rtsp
    else:
        logger.info(f"Video File: {args.video} (loop={args.loop})")
        video_source = args.video
    logger.info(f"Model: {args.model}")
    logger.info(f"Labels: {args.labels}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Confidence Threshold: {args.confidence}")

    # Initialize detector
    try:
        detector = TFLiteDetector(
            args.model,
            args.labels,
            args.confidence,
            use_tpu=not args.no_tpu
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1

    # Open video source
    logger.info("Opening video source...")
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logger.error("Failed to open video source")
        return 1

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Video source opened successfully")
    logger.info(f"Stream: {width}x{height} @ {fps:.1f} fps")

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Main detection loop
    frame_count = 0
    last_log_time = time.time()
    detection_count = 0
    fps_counter = 0
    fps_start_time = time.time()

    logger.info("Starting detection loop...")

    while True:
        ret, frame = cap.read()

        if not ret:
            if args.video and args.loop:
                logger.info("Reached end of video, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            elif args.video:
                logger.info("Reached end of video, stopping...")
                break
            else:
                logger.warning("Failed to read frame, reconnecting...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
                continue

        frame_count += 1
        fps_counter += 1

        # Run detection
        detections = detector.detect(frame)

        if detections:
            detection_count += len(detections)
            for det in detections:
                logger.info(f"Frame {frame_count}: {det['label']} "
                          f"({det['confidence']:.1%}) at ({det['x']}, {det['y']})")

        # Write detections to JSON
        write_detections_json(detections, output_path, frame_count)

        # Calculate and log FPS
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            detection_fps = fps_counter / (current_time - fps_start_time)
            logger.info(f"Detection FPS: {detection_fps:.1f}")
            fps_counter = 0
            fps_start_time = current_time

        # Periodic status log
        if current_time - last_log_time >= 30:
            logger.info(f"Status: {frame_count} frames processed, {detection_count} detections")
            last_log_time = current_time

    cap.release()
    logger.info("Detection stopped")
    return 0


if __name__ == "__main__":
    exit(main())
