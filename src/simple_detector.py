#!/usr/bin/env python3
"""
Simple Bird Detector - Snapshot-based approach for reliable bird detection

Architecture:
  Motion → Stable? → Snapshot ROI → Classify → Filter → Dedup → Log JSONL

Key features:
  - Waits for stable motion (bird landing, not flying through)
  - Classifies entire ROI (consistent framing, not MOG2 crops)
  - Deduplicates detections (same bird within time window = one log)
  - Bulletproof error handling (never crashes, always recovers)
  - Simple JSONL logging (append-only, atomic writes)
  - Health monitoring (heartbeat every 5 minutes)
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
import yaml

# Import existing bird-id modules
from adaptive_motion_detector import AdaptiveBirdDetector

try:
    from classifier import SpeciesClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    print("WARNING: Classifier not available")

try:
    from source_manager import get_source_urls, should_use_edgetpu
    SOURCE_MANAGER_AVAILABLE = True
except ImportError:
    SOURCE_MANAGER_AVAILABLE = False
    print("WARNING: Source manager not available, using defaults")

# Default paths
DEFAULT_ROOT = Path(os.environ.get("BIRDID_ROOT", "/volume1/docker/bird-id"))
DEFAULT_DATA_DIR = Path(os.environ.get("BIRDID_DATA_DIR", "/data"))
DEFAULT_CONFIG_PATH = DEFAULT_ROOT / "config.yaml"

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Configure logging to console and file."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)


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
        logger.error(f"Failed to load config: {e}")
        return {}


class StableMotionTracker:
    """
    Tracks motion stability across frames.

    A bird is considered "stable" when motion persists in roughly the same
    location for a minimum duration (e.g., bird landed and feeding).
    Flying through frame = unstable, don't trigger classification.
    """

    def __init__(self, duration_seconds: float = 1.5, max_centroid_drift: int = 100, fps: int = 5):
        """
        Args:
            duration_seconds: How long motion must persist to be "stable"
            max_centroid_drift: Maximum pixel movement allowed (centroid tracking)
            fps: Frames per second for duration calculation
        """
        self.duration_seconds = duration_seconds
        self.max_centroid_drift = max_centroid_drift
        self.min_frames = int(duration_seconds * fps)

        # Track recent motion centroids
        self.centroids = deque(maxlen=self.min_frames)
        self.last_motion_time = 0

    def reset(self):
        """Clear tracking history."""
        self.centroids.clear()
        self.last_motion_time = 0

    def update(self, has_motion: bool, centroid: Optional[Tuple[int, int]] = None) -> bool:
        """
        Update tracker with new frame.

        Args:
            has_motion: Whether motion was detected this frame
            centroid: (x, y) centroid of motion, if any

        Returns:
            True if motion is stable (ready to classify)
        """
        if not has_motion:
            # No motion - reset
            self.reset()
            return False

        if centroid is None:
            # Motion detected but no centroid provided - conservative reset
            self.reset()
            return False

        # Add centroid to history
        self.centroids.append(centroid)
        self.last_motion_time = time.time()

        # Check if we have enough frames
        if len(self.centroids) < self.min_frames:
            return False

        # Check if all centroids are close together (stable position)
        centroids_array = np.array(list(self.centroids))
        mean_centroid = centroids_array.mean(axis=0)
        distances = np.linalg.norm(centroids_array - mean_centroid, axis=1)
        max_distance = distances.max()

        if max_distance <= self.max_centroid_drift:
            logger.debug(f"Stable motion detected (drift: {max_distance:.1f}px)")
            return True

        return False


class SimpleDetector:
    """
    Simple snapshot-based bird detector.

    Detects motion, waits for stability, classifies ROI, logs results.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize detector with configuration."""
        self.config = config

        # Get detector-specific config
        detector_config = config.get('simple_detector', {})
        self.stable_duration = detector_config.get('stable_motion_duration', 1.5)
        self.debounce_seconds = detector_config.get('debounce_seconds', 3)
        self.min_confidence = detector_config.get('min_confidence', 0.7)
        self.dedup_window = detector_config.get('dedup_window_seconds', 5)
        self.save_snapshots = detector_config.get('save_snapshots', True)
        self.max_snapshots_per_species = detector_config.get('max_snapshots_per_species', 10)
        self.heartbeat_interval = detector_config.get('heartbeat_interval', 300)

        # Get motion config
        motion_config = config.get('motion', {})
        self.roi_sub = self._parse_roi(motion_config.get('roi_sub', []))
        self.roi_main = self._parse_roi(motion_config.get('roi_main', []))
        self.frame_skip = motion_config.get('frame_skip', 5)

        # Calculate effective FPS for stable motion tracking
        rtsp_fps = 30  # Assume 30fps RTSP stream
        effective_fps = rtsp_fps / (self.frame_skip + 1)

        # Motion detector and tracker will be initialized in run() after we know frame dimensions
        self.motion_detector = None
        self.motion_tracker = None
        self.effective_fps = int(effective_fps)
        self.motion_config = motion_config

        # Initialize classifier
        if not CLASSIFIER_AVAILABLE:
            raise RuntimeError("Classifier not available - cannot run simple detector")

        logger.info("Initializing classifier...")
        use_edgetpu = should_use_edgetpu() if SOURCE_MANAGER_AVAILABLE else True
        self.classifier = SpeciesClassifier(
            use_edgetpu=use_edgetpu,
            model_path=DEFAULT_DATA_DIR / "models" / "birds_v1_edgetpu.tflite",
            labels_path=DEFAULT_DATA_DIR / "models" / "inat_bird_labels.txt"
        )

        # Deduplication tracking: {species: timestamp}
        self.recent_detections = {}

        # Snapshot tracking: {species: count}
        self.snapshot_counts = {}
        self.detections_dir = DEFAULT_DATA_DIR / "detections"
        self.snapshots_dir = self.detections_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections_logged': 0,
            'errors': 0,
            'started_at': datetime.utcnow().isoformat()
        }
        self.last_heartbeat = time.time()

        logger.info(f"Simple detector initialized:")
        logger.info(f"  Stable motion: {self.stable_duration}s")
        logger.info(f"  Debounce: {self.debounce_seconds}s")
        logger.info(f"  Min confidence: {self.min_confidence:.0%}")
        logger.info(f"  Dedup window: {self.dedup_window}s")
        logger.info(f"  EdgeTPU: {use_edgetpu}")

    def _parse_roi(self, roi_points: list) -> Optional[Tuple[int, int, int, int]]:
        """Parse ROI points [[x1,y1], [x2,y2], ...] to (x, y, w, h)."""
        if not roi_points or len(roi_points) < 2:
            return None

        points = np.array(roi_points)
        x, y = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        w, h = x2 - x, y2 - y
        return (int(x), int(y), int(w), int(h))

    def _calculate_centroid(self, boxes: list) -> Optional[Tuple[int, int]]:
        """Calculate centroid of all motion boxes."""
        if not boxes:
            return None

        # Average all box centers
        centroids = []
        for box in boxes:
            x, y, w, h = box
            cx = x + w // 2
            cy = y + h // 2
            centroids.append((cx, cy))

        if not centroids:
            return None

        centroids_array = np.array(centroids)
        mean_centroid = centroids_array.mean(axis=0)
        return (int(mean_centroid[0]), int(mean_centroid[1]))

    def _is_recent_detection(self, species: str) -> bool:
        """Check if species was detected recently (within dedup window)."""
        if species not in self.recent_detections:
            return False

        elapsed = time.time() - self.recent_detections[species]
        return elapsed < self.dedup_window

    def _log_detection(self, species: str, confidence: float, snapshot_path: Optional[str] = None):
        """Append detection to today's JSONL log file."""
        # Build detection record
        detection = {
            'ts': datetime.utcnow().isoformat() + 'Z',
            'species': species,
            'conf': round(confidence, 3)
        }

        if snapshot_path:
            detection['snapshot'] = snapshot_path

        # Get today's log file
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = self.detections_dir / f"{today}.jsonl"

        # Atomic write: temp file + rename
        temp_file = log_file.with_suffix('.jsonl.tmp')
        try:
            with open(temp_file, 'a') as f:
                f.write(json.dumps(detection) + '\n')
            temp_file.replace(log_file)

            self.stats['detections_logged'] += 1
            logger.info(f"✓ Logged: {species} ({confidence:.1%})")

        except Exception as e:
            logger.error(f"Failed to log detection: {e}")
            self.stats['errors'] += 1
            # Don't crash - continue running

    def _log_heartbeat(self):
        """Log health status."""
        heartbeat = {
            'heartbeat': datetime.utcnow().isoformat() + 'Z',
            'frames': self.stats['frames_processed'],
            'detections': self.stats['detections_logged'],
            'errors': self.stats['errors'],
            'uptime_seconds': int(time.time() - time.mktime(
                datetime.fromisoformat(self.stats['started_at']).timetuple()
            ))
        }

        # Get today's log file
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = self.detections_dir / f"{today}.jsonl"

        # Atomic write
        temp_file = log_file.with_suffix('.jsonl.tmp')
        try:
            with open(temp_file, 'a') as f:
                f.write(json.dumps(heartbeat) + '\n')
            temp_file.replace(log_file)

            logger.info(f"Heartbeat: {self.stats['frames_processed']} frames, "
                       f"{self.stats['detections_logged']} detections, "
                       f"{self.stats['errors']} errors")
        except Exception as e:
            logger.error(f"Failed to log heartbeat: {e}")

    def _save_snapshot(self, frame: np.ndarray, species: str) -> Optional[str]:
        """Save snapshot if under limit for this species."""
        if not self.save_snapshots:
            return None

        # Check if we've reached the limit for this species
        current_count = self.snapshot_counts.get(species, 0)
        if current_count >= self.max_snapshots_per_species:
            return None

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_species = species.replace(' ', '_').replace('-', '_')
        filename = f"{timestamp}_{safe_species}.jpg"
        filepath = self.snapshots_dir / filename

        try:
            cv2.imwrite(str(filepath), frame)
            self.snapshot_counts[species] = current_count + 1
            logger.debug(f"Saved snapshot: {filename}")
            return f"snapshots/{filename}"
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def run(self, substream_url: str, mainstream_url: str):
        """
        Main detection loop.

        Args:
            substream_url: Lower resolution stream for motion detection
            mainstream_url: Higher resolution stream for classification
        """
        logger.info(f"Opening streams...")
        logger.info(f"  Substream (motion): {substream_url}")
        logger.info(f"  Mainstream (classification): {mainstream_url}")

        # Open video captures
        sub_cap = cv2.VideoCapture(substream_url, cv2.CAP_FFMPEG)
        main_cap = cv2.VideoCapture(mainstream_url, cv2.CAP_FFMPEG)

        if not sub_cap.isOpened() or not main_cap.isOpened():
            raise RuntimeError("Failed to open video streams")

        # Detect if we're in video mode
        is_video_mode = not substream_url.startswith("rtsp://")
        if is_video_mode:
            logger.info("VIDEO MODE: Will loop video file continuously")

        # Get frame dimensions
        sub_width = int(sub_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sub_height = int(sub_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Substream resolution: {sub_width}x{sub_height}")

        # Initialize motion detector now that we have dimensions
        min_area = self.motion_config.get('min_area', 500)
        max_area = int(sub_width * sub_height * 0.25)  # Max 25% of frame

        # Select appropriate ROI based on resolution
        # Use roi_main for 1920x1080, roi_sub for 640x360
        if sub_width >= 1920:
            roi_mask = self.roi_main
            logger.info("Using roi_main for 1920x1080 resolution")
        else:
            roi_mask = self.roi_sub
            logger.info("Using roi_sub for lower resolution")

        roi_points = None
        if roi_mask:
            # Convert back to points for AdaptiveBirdDetector
            x, y, w, h = roi_mask
            roi_points = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
            logger.info(f"ROI: {roi_points}")

        logger.info("Initializing motion detector...")
        self.motion_detector = AdaptiveBirdDetector(
            sub_width, sub_height,
            mask=roi_points,
            initial_min_area=min_area,
            initial_max_area=max_area
        )

        # Initialize stable motion tracker
        self.motion_tracker = StableMotionTracker(
            duration_seconds=self.stable_duration,
            max_centroid_drift=100,
            fps=self.effective_fps
        )

        logger.info("Starting detection loop...")
        frame_count = 0

        try:
            while True:
                # Read frames
                ret_sub, sub_frame = sub_cap.read()
                ret_main, main_frame = main_cap.read()

                # Handle end of stream
                if not ret_sub or not ret_main:
                    if is_video_mode:
                        logger.info(f"Video loop completed at frame {frame_count}, restarting...")
                        sub_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        main_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_count = 0
                        self.motion_tracker.reset()
                        continue
                    else:
                        logger.warning("Stream read failure, reconnecting...")
                        time.sleep(1)
                        sub_cap.release()
                        main_cap.release()
                        sub_cap = cv2.VideoCapture(substream_url, cv2.CAP_FFMPEG)
                        main_cap = cv2.VideoCapture(mainstream_url, cv2.CAP_FFMPEG)
                        continue

                frame_count += 1

                # Heartbeat monitoring
                if time.time() - self.last_heartbeat >= self.heartbeat_interval:
                    self._log_heartbeat()
                    self.last_heartbeat = time.time()

                # Frame skip (process every Nth frame)
                if frame_count % (self.frame_skip + 1) != 0:
                    continue

                self.stats['frames_processed'] += 1

                # Detect motion on substream
                boxes = self.motion_detector.update(sub_frame)
                has_motion = boxes is not None and len(boxes) > 0

                # Log motion activity every 30 frames for debugging
                if frame_count % 30 == 0:
                    logger.info(f"Frame {frame_count}: Motion boxes: {len(boxes) if boxes else 0}")

                # Calculate motion centroid
                centroid = self._calculate_centroid(boxes) if has_motion else None

                # Update stable motion tracker
                is_stable = self.motion_tracker.update(has_motion, centroid)

                if has_motion and not is_stable:
                    logger.debug(f"Motion detected but not stable yet (centroid: {centroid})")

                if not is_stable:
                    # Motion not stable yet - keep waiting
                    continue

                # Motion is stable - classify the ROI!
                logger.debug("Stable motion detected, classifying...")

                # Extract ROI from mainstream (higher resolution)
                if self.roi_main:
                    x, y, w, h = self.roi_main
                    roi_crop = main_frame[y:y+h, x:x+w]
                else:
                    # No ROI defined - use full frame
                    roi_crop = main_frame

                # Prepare ROI for classification:
                # 1. Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
                # 2. Resize to 224x224 (model input size)
                # 3. Add batch dimension (1, 224, 224, 3)
                # 4. Convert to uint8
                roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                roi_resized = cv2.resize(roi_rgb, (224, 224))
                roi_batch = roi_resized.reshape(1, 224, 224, 3).astype('uint8')

                # Classify with error handling
                try:
                    species, confidence = self.classifier.classify(roi_batch)
                except Exception as e:
                    logger.error(f"Classification failed: {e}")
                    self.stats['errors'] += 1
                    self.motion_tracker.reset()
                    continue

                # Filter low confidence and background
                if confidence < self.min_confidence:
                    logger.debug(f"Low confidence: {species} ({confidence:.1%})")
                    self.motion_tracker.reset()
                    continue

                if species.lower() == "background":
                    logger.debug(f"Filtered background classification")
                    self.motion_tracker.reset()
                    continue

                # Check deduplication
                if self._is_recent_detection(species):
                    logger.debug(f"Duplicate detection: {species} (within {self.dedup_window}s)")
                    self.motion_tracker.reset()
                    continue

                # Valid detection! Save snapshot and log
                snapshot_path = self._save_snapshot(roi_crop, species)
                self._log_detection(species, confidence, snapshot_path)

                # Update dedup tracking
                self.recent_detections[species] = time.time()

                # Reset motion tracker
                self.motion_tracker.reset()

                # Debounce - wait before next detection
                logger.debug(f"Debouncing for {self.debounce_seconds}s...")
                time.sleep(self.debounce_seconds)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        finally:
            sub_cap.release()
            main_cap.release()
            logger.info(f"Final stats: {self.stats}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Bird Detector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                       help="Path to config.yaml")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("=== Simple Bird Detector ===")

    # Load config
    config = load_config(args.config)

    # Get source URLs
    if SOURCE_MANAGER_AVAILABLE:
        substream_url, mainstream_url = get_source_urls(config)
    else:
        # Fallback to RTSP URLs from config
        rtsp_config = config.get('rtsp', {})
        substream_url = rtsp_config.get('sub', 'rtsp://192.168.4.9:7447/5CAx1qDdOe7zoLEQ')
        mainstream_url = rtsp_config.get('main', 'rtsp://192.168.4.9:7447/tTHjLZrVgopARpu6')

    # Create and run detector
    detector = SimpleDetector(config)
    detector.run(substream_url, mainstream_url)


if __name__ == "__main__":
    main()
