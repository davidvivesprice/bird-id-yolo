#!/usr/bin/env python3
"""Live dual-stream detector with visualization output for HLS streaming.

Outputs annotated frames to stdout for piping to ffmpeg.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
import yaml

# Optional: Import classifier if available
try:
    from classifier import SpeciesClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False

# Optional: Import source manager for video/RTSP source selection
try:
    from source_manager import get_source_urls
    SOURCE_MANAGER_AVAILABLE = True
except ImportError:
    SOURCE_MANAGER_AVAILABLE = False

# Import detector
from adaptive_motion_detector import AdaptiveBirdDetector

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
        stream=sys.stderr  # Log to stderr so stdout is clean for video
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


def extract_and_prepare_roi(frame: np.ndarray, box, padding: int = 5) -> np.ndarray:
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
    roi_batch = np.expand_dims(roi_resized, axis=0)

    return roi_batch.astype(np.uint8)


def open_rtsp_stream(url: str, name: str = "stream", max_retries: int = 3) -> Optional[cv2.VideoCapture]:
    """Open RTSP stream with retry logic.

    Args:
        url: RTSP URL to open
        name: Descriptive name for logging
        max_retries: Maximum number of connection attempts

    Returns:
        VideoCapture object if successful, None otherwise
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Opening {name} (attempt {attempt + 1}/{max_retries})...")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

            if cap.isOpened():
                # Test read to verify stream is actually working
                ret, _ = cap.read()
                if ret:
                    logger.info(f"{name} opened successfully")
                    return cap
                else:
                    logger.warning(f"{name} opened but cannot read frames")
                    cap.release()
            else:
                logger.warning(f"Cannot open {name}: {url}")

        except Exception as e:
            logger.error(f"Error opening {name}: {e}")

        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    logger.error(f"Failed to open {name} after {max_retries} attempts")
    return None


def draw_detection(frame: np.ndarray, box, label: str = "", confidence: float = 0.0,
                   color=(0, 255, 0), thickness=2):
    """Draw bounding box and label on frame."""
    x, y, w, h = box

    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Draw label background
    if label:
        label_text = f"{label}: {confidence:.1%}" if confidence > 0 else label
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x, y - text_height - baseline - 5),
            (x + text_width + 5, y),
            color,
            -1
        )
        cv2.putText(
            frame, label_text,
            (x + 2, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 2
        )


def main():
    parser = argparse.ArgumentParser(description="Live bird detector with visualization")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "config.yaml")
    parser.add_argument("--classify", action="store_true", help="Enable classification")
    parser.add_argument("--model-path", type=Path,
                       default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels-path", type=Path,
                       default=DEFAULT_LABEL_PATH)
    parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--output-fps", type=int, default=15, help="Output FPS for HLS")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min confidence to show label")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--edgetpu", action="store_true", help="Use EdgeTPU acceleration")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load config
    config = load_config(args.config)

    logger.info("=== Live Bird Detector with Visualization ===")

    # Get source URLs (RTSP or video file) based on runtime configuration
    if SOURCE_MANAGER_AVAILABLE:
        substream_url, mainstream_url = get_source_urls(config)
        logger.info(f"Source: sub={substream_url}, main={mainstream_url}")
    else:
        # Fallback to RTSP if source_manager not available
        logger.warning("source_manager not available, falling back to RTSP")
        rtsp_config = config.get("rtsp", {})
        substream_url = rtsp_config.get("sub", "rtsp://192.168.4.9:7447/BFMOkPpZnsGP0FaW")
        mainstream_url = rtsp_config.get("main", "rtsp://192.168.4.9:7447/umtyoUf5rd0izEDr")
        logger.info(f"Substream: {substream_url}")
        logger.info(f"Main stream: {mainstream_url}")

    # Initialize classifier if requested
    classifier = None
    if args.classify:
        if not CLASSIFIER_AVAILABLE:
            logger.error("Classifier not available")
            sys.exit(1)

        logger.info("Initializing classifier...")
        classifier = SpeciesClassifier(args.model_path, args.labels_path, use_edgetpu=args.edgetpu)
        logger.info("Classifier ready!")

    # Open streams with retry logic
    logger.info("Opening streams...")
    sub_cap = open_rtsp_stream(substream_url, "substream")
    if sub_cap is None:
        logger.error("Failed to open substream after retries")
        sys.exit(1)

    main_cap = open_rtsp_stream(mainstream_url, "main stream")
    if main_cap is None:
        logger.error("Failed to open main stream after retries")
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

    # Scale thresholds
    resolution_scale = (sub_width / 1920) ** 2
    min_area_scaled = int(motion_config.get("initial_min_area", 50000) * resolution_scale)
    max_area_scaled = int(motion_config.get("initial_max_area", 500000) * resolution_scale)

    detector = AdaptiveBirdDetector(
        sub_width, sub_height,
        mask=roi_sub,
        initial_min_area=min_area_scaled,
        initial_max_area=max_area_scaled,
        min_aspect=0.3,
        max_aspect=3.0
    )

    logger.info(f"Detector initialized: min_area={min_area_scaled}px")
    logger.info(f"Starting live processing (frame_skip={args.frame_skip}, output_fps={args.output_fps})")
    logger.info("Writing annotated frames to stdout for ffmpeg...")

    frame_count = 0
    output_frame_interval = int(30 / args.output_fps) if args.output_fps < 30 else 1
    consecutive_failures = 0
    max_consecutive_failures = 30  # Reconnect after 30 consecutive read failures
    last_successful_read = time.time()

    # Frame corruption tracking
    corruption_count = 0
    corruption_window = 100  # Track corruption over last 100 frames
    max_corruption_rate = 0.3  # Reconnect if >30% of frames corrupted

    # Create black frames as fallback for corrupted frames
    black_sub = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)
    black_main = np.zeros((main_height, main_width, 3), dtype=np.uint8)

    while True:
        # Read from both streams
        ret_sub, sub_frame = sub_cap.read()
        ret_main, main_frame = main_cap.read()

        if not ret_sub or not ret_main:
            consecutive_failures += 1
            logger.warning(f"Stream read failure ({consecutive_failures}/{max_consecutive_failures})")

            # Check if we should attempt reconnection
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"Too many consecutive failures ({consecutive_failures}), attempting reconnection...")

                # Release old streams
                sub_cap.release()
                main_cap.release()

                # Wait before reconnecting
                time.sleep(5)

                # Attempt to reconnect
                sub_cap = open_rtsp_stream(substream_url, "substream")
                main_cap = open_rtsp_stream(mainstream_url, "main stream")

                if sub_cap is None or main_cap is None:
                    logger.error("Reconnection failed, exiting...")
                    break

                # Reset failure counter after successful reconnection
                consecutive_failures = 0
                corruption_count = 0
                logger.info("Stream reconnection successful, resuming processing...")

                # Reset detector state
                detector = AdaptiveBirdDetector(
                    sub_width, sub_height,
                    mask=roi_sub,
                    initial_min_area=min_area_scaled,
                    initial_max_area=max_area_scaled,
                    min_aspect=0.3,
                    max_aspect=3.0
                )

                continue

            # Skip this iteration but don't break
            time.sleep(0.1)
            continue

        # Successful read - validate frame data
        frame_corrupted = False

        # Validate main frame (critical for HLS output)
        if main_frame is None or main_frame.shape != (main_height, main_width, 3):
            logger.warning(f"Main frame corruption detected: invalid shape")
            frame_corrupted = True
            main_frame = black_main.copy()
        elif np.sum(main_frame) == 0:
            logger.warning(f"Main frame corruption detected: all zeros")
            frame_corrupted = True
            main_frame = black_main.copy()

        # Validate sub frame (used for detection)
        if sub_frame is None or sub_frame.shape != (sub_height, sub_width, 3):
            logger.warning(f"Sub frame corruption detected: invalid shape")
            frame_corrupted = True
            sub_frame = black_sub.copy()

        # Track corruption rate
        if frame_corrupted:
            corruption_count += 1

        # Check if corruption rate is too high (every 100 frames)
        if frame_count > 0 and frame_count % corruption_window == 0:
            corruption_rate = corruption_count / corruption_window

            if corruption_rate > max_corruption_rate:
                logger.error(f"High corruption rate detected: {corruption_rate:.1%} - triggering reconnection")

                # Release old streams
                sub_cap.release()
                main_cap.release()

                # Wait before reconnecting
                time.sleep(5)

                # Attempt to reconnect
                sub_cap = open_rtsp_stream(substream_url, "substream")
                main_cap = open_rtsp_stream(mainstream_url, "main stream")

                if sub_cap is None or main_cap is None:
                    logger.error("Reconnection failed, exiting...")
                    break

                # Reset counters
                consecutive_failures = 0
                corruption_count = 0
                logger.info("Stream reconnection successful after high corruption, resuming processing...")

                # Reset detector state
                detector = AdaptiveBirdDetector(
                    sub_width, sub_height,
                    mask=roi_sub,
                    initial_min_area=min_area_scaled,
                    initial_max_area=max_area_scaled,
                    min_aspect=0.3,
                    max_aspect=3.0
                )

                continue
            else:
                logger.info(f"Corruption rate over last {corruption_window} frames: {corruption_rate:.1%}")
                corruption_count = 0  # Reset for next window

        # Successful read - reset failure counter
        if consecutive_failures > 0:
            logger.info(f"Stream recovered after {consecutive_failures} failures")
            consecutive_failures = 0
        last_successful_read = time.time()

        # DETECTION DISABLED FOR TESTING - Just passthrough raw camera feed
        # annotated_frame = main_frame.copy()
        #
        # if frame_count % (args.frame_skip + 1) == 0:
        #     boxes = detector.update(sub_frame)
        #
        #     if boxes:
        #         logger.debug(f"Frame {frame_count}: {len(boxes)} detections")
        #
        #         for box in boxes:
        #             # Scale to main stream
        #             main_box = scale_roi(box, sub_width, main_width)
        #
        #             if classifier:
        #                 try:
        #                     roi = extract_and_prepare_roi(main_frame, main_box, padding=5)
        #                     species, confidence = classifier.classify(roi)
        #
        #                     # Only show if above confidence threshold
        #                     if confidence >= args.min_confidence:
        #                         species_short = species.split('(')[0].strip()
        #                         draw_detection(annotated_frame, main_box, species_short, confidence,
        #                                      color=(0, 255, 0), thickness=3)
        #                         logger.info(f"  → {species_short}: {confidence:.1%}")
        #                     else:
        #                         # Draw box without label for low confidence
        #                         draw_detection(annotated_frame, main_box, "", 0,
        #                                      color=(100, 100, 100), thickness=2)
        #                 except Exception as e:
        #                     logger.error(f"Classification error: {e}")
        #                     draw_detection(annotated_frame, main_box, "Error", 0,
        #                                  color=(0, 0, 255), thickness=2)
        #             else:
        #                 # No classifier, just draw motion boxes
        #                 draw_detection(annotated_frame, main_box, "Motion", 0,
        #                              color=(255, 0, 0), thickness=2)

        # Output frame at reduced FPS - RAW PASSTHROUGH (no annotation)
        if frame_count % output_frame_interval == 0:
            # Output raw BGR24 video bytes for ffmpeg
            try:
                sys.stdout.buffer.write(main_frame.tobytes())
                sys.stdout.buffer.flush()
            except Exception as e:
                logger.error(f"Failed to write frame to stdout: {e}")

        frame_count += 1

    sub_cap.release()
    main_cap.release()
    logger.info("Stream processing completed")


if __name__ == "__main__":
    main()
