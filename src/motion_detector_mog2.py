#!/usr/bin/env python3
"""Enhanced motion detector with MOG2 background subtraction support."""
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

Mask = List[Tuple[int, int]]

# Default config path
DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"
DEFAULT_SOURCE = os.getenv("BIRD_RTSP", "rtsp://192.168.4.9:7447/tTHjLZrVgopARpu6")

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

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Format string
    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if log_config.get("console", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_config.get("file", True):
        log_dir = Path(config.get("paths", {}).get("logs", "/volume1/docker/bird-id/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "motion_detector.log"

        max_bytes = log_config.get("max_bytes", 10485760)  # 10MB
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


def extract_and_prepare_roi(frame: np.ndarray, box: Tuple[int, int, int, int],
                             target_size: Tuple[int, int] = (224, 224),
                             padding: int = 20) -> np.ndarray:
    """
    Extract ROI from frame and prepare for classification.

    Args:
        frame: Original frame
        box: Bounding box (x, y, w, h)
        target_size: Target size for classifier (width, height)
        padding: Additional pixels around box

    Returns:
        Preprocessed image ready for classification (1, H, W, 3) uint8
    """
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

    # Resize to target size
    roi_resized = cv2.resize(roi_rgb, target_size)

    # Add batch dimension
    roi_batch = np.expand_dims(roi_resized, axis=0)

    return roi_batch.astype(np.uint8)


class MotionDetector:
    """Motion detection with support for frame differencing and MOG2."""

    def __init__(self, width: int, height: int, mask: Optional[Mask] = None,
                 method: str = 'frame_diff', history: int = 30,
                 diff_threshold: int = 15, min_area: int = 500,
                 mog2_var_threshold: int = 16, detect_shadows: bool = True):
        """
        Initialize motion detector.

        Args:
            width: Frame width
            height: Frame height
            mask: Optional ROI mask coordinates
            method: 'frame_diff' or 'mog2'
            history: Number of frames for background model
            diff_threshold: Threshold for frame differencing
            min_area: Minimum contour area (pixels)
            mog2_var_threshold: Variance threshold for MOG2
            detect_shadows: Enable shadow detection in MOG2
        """
        self.method = method
        self.mask = self._build_mask(mask, width, height)
        self.min_area = min_area

        if method == 'mog2':
            # MOG2 background subtractor
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=mog2_var_threshold,
                detectShadows=detect_shadows
            )
            logger.info(f"Using MOG2 method (history={history}, varThreshold={mog2_var_threshold}, shadows={detect_shadows})")
        else:
            # Frame differencing
            self.background = None
            self.alpha = 1.0 / history
            self.diff_threshold = diff_threshold
            logger.info(f"Using frame differencing (history={history}, threshold={diff_threshold})")

    def _build_mask(self, mask: Optional[Mask], width: int, height: int):
        if not mask:
            return None
        poly = np.array([mask], dtype=np.int32)
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(canvas, poly, 255)
        return canvas

    def update(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Process frame and return bounding boxes of motion regions."""
        if self.method == 'mog2':
            return self._update_mog2(frame)
        else:
            return self._update_frame_diff(frame)

    def _update_mog2(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """MOG2 background subtraction method."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Apply ROI mask if specified
        if self.mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=self.mask)

        # Remove shadows (they're marked as 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

        return boxes

    def _update_frame_diff(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Original frame differencing method."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

        if self.mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=self.mask)

        if self.background is None:
            self.background = gray.astype("float")
            return []

        cv2.accumulateWeighted(gray, self.background, self.alpha)
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

        return boxes


def run(source: str, output_dir: Path, frames: int, mask: Optional[Mask],
        method: str, diff_threshold: int, min_area: int, history: int = 30,
        mog2_var_threshold: int = 16, detect_shadows: bool = True, frame_skip: int = 0,
        roi_main: Optional[Mask] = None, roi_sub: Optional[Mask] = None,
        classifier: Optional['SpeciesClassifier'] = None):
    """Run motion detection on video source and save debug frames with optional classification."""
    logger.info(f"Starting motion detection on source: {source}")
    logger.info(f"Method: {method}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parameters: diff_threshold={diff_threshold}, min_area={min_area}, history={history}")
    if frame_skip > 0:
        effective_fps = 30.0 / (frame_skip + 1)
        logger.info(f"Frame skip: {frame_skip} (processing every {frame_skip + 1} frames, ~{effective_fps:.1f} fps)")
    if method == 'mog2':
        logger.info(f"MOG2 params: varThreshold={mog2_var_threshold}, detectShadows={detect_shadows}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Opening video source with OpenCV")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"Unable to open source: {source}")
        raise RuntimeError(f"Unable to open source: {source}")

    ret, frame = cap.read()
    if not ret:
        logger.error("Unable to read initial frame")
        raise RuntimeError("Unable to read initial frame")

    height, width = frame.shape[:2]
    logger.info(f"Video dimensions: {width}x{height}")

    # Auto-select ROI based on resolution
    if roi_main is not None and roi_sub is not None:
        if width == 1920 and height == 1080:
            mask = roi_main
            logger.info(f"Auto-selected roi_main for 1920x1080: {mask}")
        elif width == 640 and height == 360:
            mask = roi_sub
            logger.info(f"Auto-selected roi_sub for 640x360: {mask}")
        else:
            logger.warning(f"Unexpected resolution {width}x{height}, using roi_main as fallback")
            mask = roi_main

    if mask:
        logger.info(f"Using ROI mask with {len(mask)} points")

    detector = MotionDetector(
        width, height, mask,
        method=method,
        history=history,
        diff_threshold=diff_threshold,
        min_area=min_area,
        mog2_var_threshold=mog2_var_threshold,
        detect_shadows=detect_shadows
    )

    frame_id = 0
    processed_frames = 0
    motion_frames = 0
    total_boxes = 0
    max_frames = "all" if frames <= 0 else frames

    logger.info(f"Processing {max_frames} frames")

    while frames <= 0 or processed_frames < frames:
        ret, frame = cap.read()
        if not ret:
            logger.debug(f"End of stream at frame {frame_id}")
            break

        # Only process every Nth frame based on frame_skip
        should_process = (frame_skip == 0) or (frame_id % (frame_skip + 1) == 0)

        boxes = []
        classifications = []
        if should_process:
            boxes = detector.update(frame)
            if boxes:
                motion_frames += 1
                total_boxes += len(boxes)
                logger.debug(f"Frame {frame_id}: detected {len(boxes)} motion region(s)")

                # Classify each detected bird ROI
                if classifier:
                    for box in boxes:
                        try:
                            roi = extract_and_prepare_roi(frame, box)
                            species, confidence = classifier.classify(roi)
                            classifications.append((species, confidence))
                            logger.info(f"Frame {frame_id} - Detected: {species} (confidence: {confidence:.2%})")
                        except Exception as e:
                            logger.error(f"Classification failed for box {box}: {e}")
                            classifications.append(("error", 0.0))

            processed_frames += 1

        debug = frame.copy()
        for idx, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add species label if classified
            if classifier and idx < len(classifications):
                species, confidence = classifications[idx]
                label = f"{species.split('(')[0].strip()}: {confidence:.1%}"
                cv2.putText(debug, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = output_dir / f"frame_{frame_id:04d}.jpg"
        cv2.imwrite(str(output_path), debug)
        frame_id += 1

        # Progress logging every 50 processed frames
        if processed_frames > 0 and processed_frames % 50 == 0:
            logger.info(f"Progress: {processed_frames} frames processed ({frame_id} total read), {motion_frames} with motion")

    cap.release()

    # Final summary
    motion_pct = (motion_frames / processed_frames * 100) if processed_frames > 0 else 0
    logger.info(f"=== Motion Detection Complete ===")
    if frame_skip > 0:
        logger.info(f"Total frames read: {frame_id}")
        logger.info(f"Frames processed: {processed_frames} (every {frame_skip + 1} frames)")
    else:
        logger.info(f"Frames processed: {processed_frames}")
    logger.info(f"Motion detected: {motion_frames} frames ({motion_pct:.1f}%)")
    logger.info(f"Total bounding boxes: {total_boxes}")
    logger.info(f"Debug frames saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run motion detection on RTSP stream or video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to config.yaml (default: ../config.yaml)")
    parser.add_argument("--source", help="RTSP URL or path to video clip (overrides config)")
    parser.add_argument("--output", help="Where to write debug frames (overrides config)")
    parser.add_argument("--frames", type=int, help="How many frames to process (<=0 for all)")
    parser.add_argument("--mask", action="store_true", help="Use ROI mask from config")
    parser.add_argument("--method", choices=["frame_diff", "mog2"], help="Detection method (overrides config)")
    parser.add_argument("--diff-threshold", type=int, help="Pixel difference threshold (overrides config)")
    parser.add_argument("--min-area", type=int, help="Minimum contour area to keep (overrides config)")
    parser.add_argument("--history", type=int, help="Background history frames (overrides config)")
    parser.add_argument("--mog2-var-threshold", type=int, help="MOG2 variance threshold (overrides config)")
    parser.add_argument("--detect-shadows", type=bool, help="MOG2 shadow detection (overrides config)")
    parser.add_argument("--frame-skip", type=int, help="Process every Nth frame (0=all, 5=every 6th for ~5fps)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (overrides config)")

    # Classification arguments
    parser.add_argument("--classify", action="store_true", help="Enable bird species classification")
    parser.add_argument("--model-path", type=Path,
                       default=Path("/volume1/docker/bird-id/data/models/birds_v1.tflite"),
                       help="Path to TFLite classification model")
    parser.add_argument("--labels-path", type=Path,
                       default=Path("/volume1/docker/bird-id/data/models/inat_bird_labels.txt"),
                       help="Path to species labels file")
    parser.add_argument("--edgetpu", action="store_true", help="Use EdgeTPU acceleration for classification")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging (do this early)
    if args.log_level:
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["level"] = args.log_level
    setup_logging(config)

    logger.info("=== Bird-ID Motion Detector (with MOG2 support) ===")

    # Get motion detection parameters from config with CLI overrides
    motion_config = config.get("motion", {})
    rtsp_config = config.get("rtsp", {})
    paths_config = config.get("paths", {})

    # Source (CLI > env > config > default)
    source = args.source or os.getenv("BIRD_RTSP") or rtsp_config.get("main") or DEFAULT_SOURCE

    # Output directory (CLI > config > default)
    output = args.output or paths_config.get("debug", "/volume1/docker/bird-id/data/debug")

    # Detection method
    method = args.method if args.method is not None else motion_config.get("method", "frame_diff")

    # Processing parameters (CLI overrides config)
    frames = args.frames if args.frames is not None else 200
    diff_threshold = args.diff_threshold if args.diff_threshold is not None else motion_config.get("diff_threshold", 15)
    min_area = args.min_area if args.min_area is not None else motion_config.get("min_area", 500)
    history = args.history if args.history is not None else motion_config.get("history", 30)

    # MOG2 specific parameters
    mog2_var_threshold = args.mog2_var_threshold if args.mog2_var_threshold is not None else motion_config.get("mog2_var_threshold", 16)
    detect_shadows = args.detect_shadows if args.detect_shadows is not None else motion_config.get("detect_shadows", True)

    # Frame skip parameter (0=process all frames, 5=every 6th frame for ~5fps)
    frame_skip = args.frame_skip if args.frame_skip is not None else motion_config.get("frame_skip", 0)

    # ROI mask - collect both resolutions for auto-selection
    mask = None
    roi_main = None
    roi_sub = None

    if args.mask:
        # Load ROI coordinates for both resolutions
        roi_main = motion_config.get("roi_main") or [(380, 425), (1550, 425), (1550, 960), (380, 960)]
        roi_sub = motion_config.get("roi_sub") or [(127, 142), (517, 142), (517, 320), (127, 320)]
        logger.info("ROI mask enabled - will auto-select based on resolution")
    elif motion_config.get("roi"):
        # Backward compatibility: if only "roi" is specified, use it directly
        mask = motion_config.get("roi")
        logger.info(f"Using legacy ROI mask: {mask}")

    # Initialize classifier if requested
    classifier = None
    if args.classify:
        if not CLASSIFIER_AVAILABLE:
            logger.error("Classification requested but classifier module not available")
            sys.exit(1)

        if not args.model_path.exists():
            logger.error(f"Model file not found: {args.model_path}")
            sys.exit(1)

        if not args.labels_path.exists():
            logger.error(f"Labels file not found: {args.labels_path}")
            sys.exit(1)

        logger.info(f"Initializing bird classifier...")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Labels: {args.labels_path}")
        logger.info(f"EdgeTPU: {args.edgetpu}")

        try:
            classifier = SpeciesClassifier(
                args.model_path,
                args.labels_path,
                use_edgetpu=args.edgetpu
            )
            logger.info("Classifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            sys.exit(1)

    try:
        run(str(source), Path(output), frames, mask, method,
            diff_threshold=diff_threshold, min_area=min_area, history=history,
            mog2_var_threshold=mog2_var_threshold, detect_shadows=detect_shadows,
            frame_skip=frame_skip, roi_main=roi_main, roi_sub=roi_sub,
            classifier=classifier)
    except Exception as e:
        logger.exception(f"Motion detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
