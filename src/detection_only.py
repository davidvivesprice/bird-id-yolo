#!/usr/bin/env python3
"""Detection-only script - outputs coordinates, no video encoding."""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
import cv2
import yaml

from adaptive_motion_detector import AdaptiveBirdDetector

try:
    from classifier import SpeciesClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False

try:
    from source_manager import get_source_urls
    SOURCE_MANAGER_AVAILABLE = True
except ImportError:
    SOURCE_MANAGER_AVAILABLE = False

DEFAULT_ROOT = Path(os.environ.get("BIRDID_ROOT", "/volume1/docker/bird-id"))
DEFAULT_DATA_DIR = Path(os.environ.get("BIRDID_DATA_DIR", "/data"))
DEFAULT_MODEL_PATH = Path(os.environ.get("BIRDID_MODEL_PATH", "/data/models/birds_v1_edgetpu.tflite"))
DEFAULT_LABEL_PATH = Path(os.environ.get("BIRDID_LABELS_PATH", "/data/models/inat_bird_labels.txt"))

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )


def scale_roi(box, from_width: int, to_width: int):
    """Scale bounding box from one resolution to another."""
    x, y, w, h = box
    scale = to_width / from_width
    return (
        int(x * scale),
        int(y * scale),
        int(w * scale),
        int(h * scale)
    )


def extract_and_prepare_roi(frame, box, padding: int = 5):
    """Extract ROI from frame and prepare for classification."""
    x, y, w, h = box
    height, width = frame.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (224, 224))
    return roi_resized


def main():
    parser = argparse.ArgumentParser(description="Detection-only - outputs coordinates")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "config.yaml")
    parser.add_argument("--classify", action="store_true", help="Enable classification")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min confidence threshold")
    parser.add_argument("--output-file", type=Path, default="/share/detections.json", help="Detection output file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--edgetpu", action="store_true", help="Use EdgeTPU acceleration")
    parser.add_argument("--persistence-frames", type=int, default=5, help="Keep detections for N frames")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load config
    config = load_config(args.config)
    logger.info("=== Detection-Only Process ===")

    # Get source URLs
    if SOURCE_MANAGER_AVAILABLE:
        substream_url, mainstream_url = get_source_urls(config)
    else:
        rtsp_config = config.get("rtsp", {})
        substream_url = rtsp_config.get("sub", "rtsp://192.168.4.9:7447/BFMOkPpZnsGP0FaW")
        mainstream_url = rtsp_config.get("main", "rtsp://192.168.4.9:7447/umtyoUf5rd0izEDr")

    # Detect if we're in video mode (file paths vs rtsp://)
    is_video_mode = not substream_url.startswith("rtsp://")

    logger.info(f"Substream: {substream_url}")
    logger.info(f"Main stream: {mainstream_url}")
    if is_video_mode:
        logger.info("VIDEO MODE: Will loop video file continuously for testing")

    # Initialize classifier if requested
    classifier = None
    if args.classify:
        if not CLASSIFIER_AVAILABLE:
            logger.error("Classifier not available")
            sys.exit(1)
        logger.info("Initializing classifier...")
        classifier = SpeciesClassifier(args.model_path, args.labels_path, use_edgetpu=args.edgetpu)
        logger.info("Classifier ready!")

    # Open streams
    logger.info("Opening streams...")
    sub_cap = cv2.VideoCapture(substream_url, cv2.CAP_FFMPEG)
    main_cap = cv2.VideoCapture(mainstream_url, cv2.CAP_FFMPEG)

    if not sub_cap.isOpened() or not main_cap.isOpened():
        logger.error("Failed to open streams")
        sys.exit(1)

    # Get dimensions
    ret, sub_frame = sub_cap.read()
    ret, main_frame = main_cap.read()

    sub_height, sub_width = sub_frame.shape[:2]
    main_height, main_width = main_frame.shape[:2]

    logger.info(f"Substream: {sub_width}x{sub_height}")
    logger.info(f"Main stream: {main_width}x{main_height}")

    # Initialize detector
    motion_config = config.get("motion", {})
    roi_sub = motion_config.get("roi_sub", [(127, 142), (517, 142), (517, 320), (127, 320)])

    # Use basic min_area from config instead of scaled adaptive thresholds
    # Birds can be small at 640x360, so use 500px minimum
    min_area_basic = motion_config.get("min_area", 500)
    max_area_basic = int((sub_width * sub_height) * 0.25)  # Max 25% of frame (not 50%)

    detector = AdaptiveBirdDetector(
        sub_width, sub_height,
        mask=roi_sub,  # RE-ENABLED ROI - restrict to feeder area
        initial_min_area=min_area_basic,
        initial_max_area=max_area_basic,
        min_aspect=0.3,
        max_aspect=3.0
    )

    logger.info(f"Detector initialized, starting detection loop...")

    frame_count = 0
    last_detections = []
    last_detection_frame = -999  # Track when we last had detections

    while True:
        ret_sub, sub_frame = sub_cap.read()
        ret_main, main_frame = main_cap.read()

        if not ret_sub or not ret_main:
            if is_video_mode:
                # Video file ended - loop back to start for continuous testing
                logger.info(f"Video loop completed at frame {frame_count}. Restarting from beginning...")
                sub_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                main_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0  # Reset frame counter for new loop
                continue
            else:
                # RTSP stream failure - reconnect
                logger.warning("Stream read failure, reconnecting...")
                time.sleep(1)
                sub_cap.release()
                main_cap.release()
                sub_cap = cv2.VideoCapture(substream_url, cv2.CAP_FFMPEG)
                main_cap = cv2.VideoCapture(mainstream_url, cv2.CAP_FFMPEG)
                continue

        # Detection on substream (with frame skip)
        if frame_count % (args.frame_skip + 1) == 0:
            boxes = detector.update(sub_frame)

            # DEBUG: Log raw detection count
            if frame_count % 30 == 0:  # Every 30 processed frames (~30 seconds)
                logger.info(f"Frame {frame_count}: MOG2 found {len(boxes) if boxes else 0} motion blobs")

            detections = []

            if boxes:
                # Motion detected - update persistence timer
                last_detection_frame = frame_count

                for box in boxes:
                    # Scale to main stream
                    main_box = scale_roi(box, sub_width, main_width)
                    x, y, w, h = main_box

                    detection = {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "label": "Motion",
                        "confidence": 0.0
                    }

                    if classifier:
                        try:
                            roi = extract_and_prepare_roi(main_frame, main_box, padding=5)
                            roi_batch = roi.reshape(1, 224, 224, 3).astype('uint8')
                            species, confidence = classifier.classify(roi_batch)

                            if confidence >= args.min_confidence:
                                species_short = species.split('(')[0].strip()

                                # Filter out "background" classifications
                                if species_short.lower() != "background":
                                    detection["label"] = species_short
                                    detection["confidence"] = float(confidence)
                                    logger.info(f"Detection: {species_short} ({confidence:.1%}) at ({x},{y},{w},{h})")
                                    detections.append(detection)
                                else:
                                    logger.info(f"FILTERED background ({confidence:.1%}) at ({x},{y},{w},{h})")
                            else:
                                # Below confidence threshold, skip
                                pass
                        except Exception as e:
                            logger.error(f"Classification error: {e}")

                # Update last_detections with current detections (even if empty due to filtering)
                if detections:
                    last_detections = detections
            else:
                # No motion detected - keep previous detections if within persistence window
                if frame_count - last_detection_frame > args.persistence_frames:
                    last_detections = []

            # Write detections to file (atomic write)
            try:
                output_data = {
                    "timestamp": time.time(),
                    "frame": frame_count,
                    "detections": last_detections
                }
                tmp_file = args.output_file.with_suffix('.tmp')
                with open(tmp_file, 'w') as f:
                    json.dump(output_data, f)
                tmp_file.rename(args.output_file)
            except Exception as e:
                logger.error(f"Failed to write detections: {e}")

        frame_count += 1

    sub_cap.release()
    main_cap.release()
    logger.info("Detection process completed")


if __name__ == "__main__":
    main()
