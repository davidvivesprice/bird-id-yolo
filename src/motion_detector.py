#!/usr/bin/env python3
"""Simple frame-differencing motion detector for the bird-id project."""
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


class MotionDetector:
    def __init__(self, width: int, height: int, mask: Optional[Mask] = None,
                 history: int = 30, diff_threshold: int = 15, min_area: int = 500):
        self.background = None
        self.alpha = 1.0 / history
        self.mask = self._build_mask(mask, width, height)
        self.diff_threshold = diff_threshold
        self.min_area = min_area

    def _build_mask(self, mask: Optional[Mask], width: int, height: int):
        if not mask:
            return None
        poly = np.array([mask], dtype=np.int32)
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(canvas, poly, 255)
        return canvas

    def update(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
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
        diff_threshold: int, min_area: int, history: int = 30):
    """Run motion detection on video source and save debug frames."""
    logger.info(f"Starting motion detection on source: {source}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parameters: diff_threshold={diff_threshold}, min_area={min_area}, history={history}")

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

    if mask:
        logger.info(f"Using ROI mask with {len(mask)} points")

    detector = MotionDetector(width, height, mask,
                              history=history,
                              diff_threshold=diff_threshold,
                              min_area=min_area)

    frame_id = 0
    motion_frames = 0
    total_boxes = 0
    max_frames = "all" if frames <= 0 else frames

    logger.info(f"Processing {max_frames} frames")

    while frames <= 0 or frame_id < frames:
        ret, frame = cap.read()
        if not ret:
            logger.debug(f"End of stream at frame {frame_id}")
            break

        boxes = detector.update(frame)
        if boxes:
            motion_frames += 1
            total_boxes += len(boxes)
            logger.debug(f"Frame {frame_id}: detected {len(boxes)} motion region(s)")

        debug = frame.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_path = output_dir / f"frame_{frame_id:04d}.jpg"
        cv2.imwrite(str(output_path), debug)
        frame_id += 1

        # Progress logging every 50 frames
        if frame_id % 50 == 0:
            logger.info(f"Progress: {frame_id} frames processed, {motion_frames} with motion")

    cap.release()

    # Final summary
    motion_pct = (motion_frames / frame_id * 100) if frame_id > 0 else 0
    logger.info(f"=== Motion Detection Complete ===")
    logger.info(f"Processed: {frame_id} frames")
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
    parser.add_argument("--diff-threshold", type=int, help="Pixel difference threshold (overrides config)")
    parser.add_argument("--min-area", type=int, help="Minimum contour area to keep (overrides config)")
    parser.add_argument("--history", type=int, help="Background history frames (overrides config)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (overrides config)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging (do this early)
    if args.log_level:
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["level"] = args.log_level
    setup_logging(config)

    logger.info("=== Bird-ID Motion Detector ===")

    # Get motion detection parameters from config with CLI overrides
    motion_config = config.get("motion", {})
    rtsp_config = config.get("rtsp", {})
    paths_config = config.get("paths", {})

    # Source (CLI > env > config > default)
    source = args.source or os.getenv("BIRD_RTSP") or rtsp_config.get("main") or DEFAULT_SOURCE

    # Output directory (CLI > config > default)
    output = args.output or paths_config.get("debug", "/volume1/docker/bird-id/data/debug")

    # Processing parameters (CLI overrides config)
    frames = args.frames if args.frames is not None else 200
    diff_threshold = args.diff_threshold if args.diff_threshold is not None else motion_config.get("diff_threshold", 15)
    min_area = args.min_area if args.min_area is not None else motion_config.get("min_area", 500)
    history = args.history if args.history is not None else motion_config.get("history", 30)

    # ROI mask
    mask = None
    if args.mask or motion_config.get("roi"):
        mask = motion_config.get("roi") or [(380, 425), (1550, 425), (1550, 960), (380, 960)]
        logger.info(f"Using ROI mask: {mask}")

    try:
        run(str(source), Path(output), frames, mask,
            diff_threshold=diff_threshold, min_area=min_area, history=history)
    except Exception as e:
        logger.exception(f"Motion detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
