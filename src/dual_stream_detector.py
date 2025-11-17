#!/usr/bin/env python3
"""Dual-stream bird detection and classification.

Detection on substream (640x360) for performance.
Classification on main stream (1920x1080) for accuracy.
"""
import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import yaml

# Optional: Import classifier if available
try:
    from classifier import SpeciesClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False

# Optional: Import database
try:
    from bird_database import BirdDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

Mask = List[Tuple[int, int]]

# Default config path
DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"

DEFAULT_ROOT = Path(os.environ.get("BIRDID_ROOT", "/volume1/docker/bird-id"))
DEFAULT_DATA_DIR = Path(os.environ.get("BIRDID_DATA_DIR", DEFAULT_ROOT / "data"))
DEFAULT_LOG_DIR = Path(os.environ.get("BIRDID_LOG_DIR", DEFAULT_ROOT / "logs"))
DEFAULT_MODEL_PATH = Path(os.environ.get("BIRDID_MODEL_PATH", DEFAULT_DATA_DIR / "models/birds_v1.tflite"))
DEFAULT_LABEL_PATH = Path(os.environ.get("BIRDID_LABELS_PATH", DEFAULT_DATA_DIR / "models/inat_bird_labels.txt"))
DEFAULT_DB_PATH = Path(os.environ.get("BIRDID_DB_PATH", DEFAULT_DATA_DIR / "birds.db"))

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure structured logging based on config."""
    log_config = config.get("logging", {})
    level_name = log_config.get("level", "INFO")
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if log_config.get("console", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        root_logger.addHandler(console_handler)

    if log_config.get("file", True):
        log_dir = Path(config.get("paths", {}).get("logs") or DEFAULT_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "dual_stream_detector.log"

        max_bytes = log_config.get("max_bytes", 10485760)
        backup_count = log_config.get("backup_count", 5)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)

        logger.info(f"Logging to {log_file}")


def scale_roi(box: Tuple[int, int, int, int], from_width: int, to_width: int) -> Tuple[int, int, int, int]:
    """Scale bounding box from one resolution to another."""
    x, y, w, h = box
    scale = to_width / from_width
    return (
        int(x * scale),
        int(y * scale),
        int(w * scale),
        int(h * scale)
    )


def extract_and_prepare_roi(frame: np.ndarray, box: Tuple[int, int, int, int],
                             target_size: Tuple[int, int] = (224, 224),
                             padding: int = 5) -> np.ndarray:
    """Extract ROI from frame and prepare for classification with minimal padding."""
    x, y, w, h = box
    height, width = frame.shape[:2]

    # Add minimal padding - focus on bird only
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    # Extract ROI
    roi = frame[y1:y2, x1:x2]

    # Convert BGR to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Resize to target size
    roi_resized = cv2.resize(roi_rgb, target_size)

    # Add batch dimension
    roi_batch = np.expand_dims(roi_resized, axis=0)

    return roi_batch.astype(np.uint8)


class DualStreamDetector:
    """Dual-stream bird detector with synchronized streams."""

    def __init__(self, substream_url: str, mainstream_url: str, config: Dict[str, Any],
                 classifier: Optional['SpeciesClassifier'] = None,
                 database: Optional['BirdDatabase'] = None):
        self.substream_url = substream_url
        self.mainstream_url = mainstream_url
        self.config = config
        self.classifier = classifier
        self.database = database

        motion_config = config.get("motion", {})

        # Import here to avoid circular dependency
        from motion_detector_mog2 import MotionDetector

        # Open both streams
        logger.info(f"Opening substream: {substream_url}")
        self.sub_cap = cv2.VideoCapture(substream_url, cv2.CAP_FFMPEG)
        if not self.sub_cap.isOpened():
            raise RuntimeError(f"Cannot open substream: {substream_url}")

        logger.info(f"Opening main stream: {mainstream_url}")
        self.main_cap = cv2.VideoCapture(mainstream_url, cv2.CAP_FFMPEG)
        if not self.main_cap.isOpened():
            raise RuntimeError(f"Cannot open main stream: {mainstream_url}")

        # Read first frames to get dimensions
        ret, sub_frame = self.sub_cap.read()
        if not ret:
            raise RuntimeError("Cannot read from substream")

        ret, main_frame = self.main_cap.read()
        if not ret:
            raise RuntimeError("Cannot read from main stream")

        self.sub_height, self.sub_width = sub_frame.shape[:2]
        self.main_height, self.main_width = main_frame.shape[:2]

        logger.info(f"Substream resolution: {self.sub_width}x{self.sub_height}")
        logger.info(f"Main stream resolution: {self.main_width}x{self.main_height}")

        # Initialize motion detector on substream
        roi_sub = motion_config.get("roi_sub", [(127, 142), (517, 142), (517, 320), (127, 320)])

        self.detector = MotionDetector(
            self.sub_width, self.sub_height,
            mask=roi_sub,
            method=motion_config.get("method", "mog2"),
            history=motion_config.get("history", 500),
            min_area=motion_config.get("min_area", 500),
            mog2_var_threshold=motion_config.get("mog2_var_threshold", 16),
            detect_shadows=motion_config.get("detect_shadows", True)
        )

        logger.info("Dual-stream detector initialized")

    def run(self, max_frames: int = 0, frame_skip: int = 5):
        """Run dual-stream detection and classification."""
        frame_count = 0
        detections = 0
        classifications = 0

        logger.info(f"Starting dual-stream processing (frame_skip={frame_skip})")

        while max_frames <= 0 or frame_count < max_frames:
            # Read from both streams
            ret_sub, sub_frame = self.sub_cap.read()
            ret_main, main_frame = self.main_cap.read()

            if not ret_sub or not ret_main:
                logger.info("End of stream reached")
                break

            # Process detection on substream (with frame skip)
            should_detect = (frame_skip == 0) or (frame_count % (frame_skip + 1) == 0)

            if should_detect:
                boxes = self.detector.update(sub_frame)

                if boxes:
                    detections += len(boxes)
                    logger.info(f"Frame {frame_count}: Detected {len(boxes)} bird(s)")

                    # For each detection, classify using main stream
                    for box in boxes:
                        # Scale box coordinates from substream to main stream
                        main_box = scale_roi(box, self.sub_width, self.main_width)

                        if self.classifier:
                            try:
                                # Extract high-quality ROI from main stream
                                roi = extract_and_prepare_roi(main_frame, main_box, padding=5)

                                # Classify
                                species, confidence = self.classifier.classify(roi)
                                classifications += 1

                                logger.info(f"  → {species.split('(')[0].strip()}: {confidence:.1%}")

                                # Store in database
                                if self.database:
                                    try:
                                        self.database.add_sighting(
                                            species=species,
                                            confidence=confidence,
                                            bbox=main_box,
                                            frame_size=(self.main_width, self.main_height),
                                            stream_source="mainstream"
                                        )
                                    except Exception as db_error:
                                        logger.error(f"Database storage failed: {db_error}")

                            except Exception as e:
                                logger.error(f"Classification failed: {e}")

            frame_count += 1

            # Progress logging
            if frame_count % 100 == 0:
                logger.info(f"Progress: {frame_count} frames, {detections} detections, {classifications} classified")

        # Cleanup
        self.sub_cap.release()
        self.main_cap.release()

        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Detections: {detections}")
        logger.info(f"Classifications: {classifications}")


def main():
    parser = argparse.ArgumentParser(description="Dual-stream bird detection and classification")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                       help="Path to config.yaml")
    parser.add_argument("--frames", type=int, default=0,
                       help="Max frames to process (0=unlimited)")
    parser.add_argument("--frame-skip", type=int, default=5,
                       help="Process every Nth frame")
    parser.add_argument("--classify", action="store_true",
                       help="Enable classification")
    parser.add_argument("--model-path", type=Path,
                       default=DEFAULT_MODEL_PATH,
                       help="Path to TFLite model")
    parser.add_argument("--labels-path", type=Path,
                       default=DEFAULT_LABEL_PATH,
                       help="Path to labels file")
    parser.add_argument("--edgetpu", action="store_true",
                       help="Use EdgeTPU acceleration")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(config)

    logger.info("=== Dual-Stream Bird Detector ===")

    rtsp_config = config.get("rtsp", {})
    substream_url = rtsp_config.get("sub", "rtsp://192.168.4.9:7447/5CAx1qDdOe7zoLEQ")
    mainstream_url = rtsp_config.get("main", "rtsp://192.168.4.9:7447/tTHjLZrVgopARpu6")

    # Initialize classifier if requested
    classifier = None
    if args.classify:
        if not CLASSIFIER_AVAILABLE:
            logger.error("Classifier not available")
            sys.exit(1)

        logger.info("Initializing classifier...")
        classifier = SpeciesClassifier(
            args.model_path,
            args.labels_path,
            use_edgetpu=args.edgetpu
        )
        logger.info("Classifier ready!")

    # Initialize database if classification enabled
    database = None
    if args.classify:
        if not DATABASE_AVAILABLE:
            logger.warning("Database not available - sightings will not be stored")
        else:
            db_path = Path(config.get("paths", {}).get("database") or DEFAULT_DB_PATH)
            database = BirdDatabase(db_path)
            logger.info(f"Database ready: {db_path}")

    # Run detection
    try:
        detector = DualStreamDetector(substream_url, mainstream_url, config, classifier, database)
        detector.run(max_frames=args.frames, frame_skip=args.frame_skip)
    except Exception as e:
        logger.exception(f"Detection failed: {e}")
        sys.exit(1)
    finally:
        if database:
            database.close()


if __name__ == "__main__":
    main()
