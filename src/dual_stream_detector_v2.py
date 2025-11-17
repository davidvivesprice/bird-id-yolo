#!/usr/bin/env python3
"""Dual-stream bird detection and classification with adaptive learning.

Detection on substream (640x360) for performance.
Classification on main stream (1920x1080) for accuracy.
Multi-frame classification for improved confidence.
Adaptive motion detection that learns bird sizes from classifier feedback.
"""
import argparse
import logging
import os
import sys
from collections import defaultdict
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

# Optional: Import source manager for video/RTSP source selection
try:
    from source_manager import get_source_urls
    SOURCE_MANAGER_AVAILABLE = True
except ImportError:
    SOURCE_MANAGER_AVAILABLE = False

# Optional: Import database
try:
    from bird_database import BirdDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

Mask = List[Tuple[int, int]]

# Default config path
DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"

# Environment-based defaults for Docker deployment
DEFAULT_MODEL_PATH = Path(os.environ.get("BIRDID_MODEL_PATH", "/data/models/birds_v1_edgetpu.tflite"))
DEFAULT_LABELS_PATH = Path(os.environ.get("BIRDID_LABELS_PATH", "/data/models/inat_bird_labels.txt"))

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
        log_dir = Path(config.get("paths", {}).get("logs", "/volume1/docker/bird-id/logs"))
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


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


class BirdTracker:
    """Track birds across frames and aggregate classifications."""

    def __init__(self, min_confidence: float = 0.5, max_age: int = 30, iou_threshold: float = 0.3):
        """
        Initialize bird tracker.

        Args:
            min_confidence: Minimum confidence to log bird to database
            max_age: Maximum frames without detection before finalizing bird
            iou_threshold: Minimum IoU to consider same bird
        """
        self.min_confidence = min_confidence
        self.max_age = max_age
        self.iou_threshold = iou_threshold

        self.tracked_birds = {}
        self.next_id = 0
        self.finalized_count = 0

    def update(self, boxes: List[Tuple[int, int, int, int]], frame_num: int,
               classifier, main_frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections and return finalized birds.

        Returns list of birds that should be logged to database.
        """
        finalized = []

        # Match new detections to existing tracked birds
        matched = set()
        for box in boxes:
            best_match = None
            best_iou = self.iou_threshold

            # Find best matching tracked bird
            for bird_id, bird_data in self.tracked_birds.items():
                last_box = bird_data['last_box']
                iou = calculate_iou(box, last_box)

                if iou > best_iou:
                    best_iou = iou
                    best_match = bird_id

            if best_match is not None:
                # Update existing bird
                self._update_bird(best_match, box, frame_num, classifier, main_frame)
                matched.add(best_match)
            else:
                # Create new tracked bird
                self._create_bird(box, frame_num, classifier, main_frame)

        # Age out birds that weren't matched
        to_finalize = []
        for bird_id in list(self.tracked_birds.keys()):
            if bird_id not in matched:
                self.tracked_birds[bird_id]['age'] += 1

                if self.tracked_birds[bird_id]['age'] > self.max_age:
                    to_finalize.append(bird_id)

        # Finalize old birds
        for bird_id in to_finalize:
            bird_result = self._finalize_bird(bird_id)
            if bird_result:
                finalized.append(bird_result)

        return finalized

    def _create_bird(self, box: Tuple[int, int, int, int], frame_num: int,
                     classifier, main_frame: np.ndarray):
        """Create new tracked bird."""
        bird_id = self.next_id
        self.next_id += 1

        # Classify
        species, confidence = self._classify_box(box, classifier, main_frame)

        self.tracked_birds[bird_id] = {
            'id': bird_id,
            'first_frame': frame_num,
            'last_frame': frame_num,
            'last_box': box,
            'age': 0,
            'classifications': [(species, confidence)],
            'best_classification': (species, confidence),
        }

        logger.debug(f"Created bird {bird_id}: {species.split('(')[0].strip()} ({confidence:.1%})")

    def _update_bird(self, bird_id: int, box: Tuple[int, int, int, int],
                     frame_num: int, classifier, main_frame: np.ndarray):
        """Update existing tracked bird with new detection."""
        bird = self.tracked_birds[bird_id]

        # Classify new detection
        species, confidence = self._classify_box(box, classifier, main_frame)

        # Update bird data
        bird['last_frame'] = frame_num
        bird['last_box'] = box
        bird['age'] = 0
        bird['classifications'].append((species, confidence))

        # Update best classification if this one is better
        best_species, best_conf = bird['best_classification']
        if confidence > best_conf:
            bird['best_classification'] = (species, confidence)
            logger.debug(f"Bird {bird_id}: New best {species.split('(')[0].strip()} ({confidence:.1%})")

    def _classify_box(self, box: Tuple[int, int, int, int], classifier, main_frame: np.ndarray) -> Tuple[str, float]:
        """Classify a bounding box."""
        try:
            roi = extract_and_prepare_roi(main_frame, box, padding=5)
            return classifier.classify(roi)
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ("unknown", 0.0)

    def _finalize_bird(self, bird_id: int) -> Optional[Dict[str, Any]]:
        """Finalize a tracked bird and return result if meets threshold."""
        bird = self.tracked_birds.pop(bird_id)
        species, confidence = bird['best_classification']

        num_frames = bird['last_frame'] - bird['first_frame'] + 1
        num_classifications = len(bird['classifications'])

        logger.info(f"Finalized bird {bird_id}: {species.split('(')[0].strip()} "
                   f"({confidence:.1%} best of {num_classifications} classifications, "
                   f"{num_frames} frames)")

        self.finalized_count += 1

        # Only return if meets confidence threshold AND is not background
        if confidence >= self.min_confidence and 'background' not in species.lower():
            return {
                'species': species,
                'confidence': confidence,
                'bbox': bird['last_box'],
                'num_classifications': num_classifications,
                'num_frames': num_frames
            }
        else:
            if 'background' in species.lower():
                logger.debug(f"Bird {bird_id} classified as background, not logging")
            else:
                logger.debug(f"Bird {bird_id} below confidence threshold ({confidence:.1%} < {self.min_confidence:.1%})")
            return None

    def finalize_all(self) -> List[Dict[str, Any]]:
        """Finalize all remaining tracked birds."""
        finalized = []
        for bird_id in list(self.tracked_birds.keys()):
            bird_result = self._finalize_bird(bird_id)
            if bird_result:
                finalized.append(bird_result)
        return finalized


class DualStreamDetector:
    """Dual-stream bird detector with synchronized streams and multi-frame tracking."""

    def __init__(self, substream_url: str, mainstream_url: str, config: Dict[str, Any],
                 classifier: Optional['SpeciesClassifier'] = None,
                 database: Optional['BirdDatabase'] = None,
                 min_confidence: float = 0.5,
                 tracker_max_age: int = 30):
        self.substream_url = substream_url
        self.mainstream_url = mainstream_url
        self.config = config
        self.classifier = classifier
        self.database = database

        motion_config = config.get("motion", {})

        # Import here to avoid circular dependency
        from adaptive_motion_detector import AdaptiveBirdDetector

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

        # Initialize adaptive motion detector on substream
        roi_sub = motion_config.get("roi_sub", [(127, 142), (517, 142), (517, 320), (127, 320)])

        # Scale thresholds for substream resolution (calibrated for 1920x1080)
        # At 640x360: scale factor = (640/1920)^2 = 0.111
        resolution_scale = (self.sub_width / 1920) ** 2
        min_area_1080p = motion_config.get("initial_min_area", 50000)
        max_area_1080p = motion_config.get("initial_max_area", 500000)

        min_area_scaled = int(min_area_1080p * resolution_scale)
        max_area_scaled = int(max_area_1080p * resolution_scale)

        logger.info(f"Resolution scaling: {self.sub_width}x{self.sub_height} -> scale={resolution_scale:.3f}")
        logger.info(f"Scaled thresholds: min_area={min_area_scaled}px (from {min_area_1080p}px @ 1080p)")

        # Adaptive detector learns bird sizes from classifier feedback
        self.detector = AdaptiveBirdDetector(
            self.sub_width, self.sub_height,
            mask=roi_sub,
            initial_min_area=min_area_scaled,
            initial_max_area=max_area_scaled,
            min_aspect=motion_config.get("min_aspect", 0.3),  # Filter thin streaks
            max_aspect=motion_config.get("max_aspect", 3.0),
            learning_window=motion_config.get("learning_window", 100)
        )

        # Initialize bird tracker
        self.tracker = BirdTracker(
            min_confidence=min_confidence,
            max_age=tracker_max_age
        )

        logger.info(f"Multi-frame tracking enabled (min_confidence={min_confidence:.1%}, max_age={tracker_max_age})")
        logger.info(f"Adaptive learning enabled (starts at min_area={self.detector.min_area}px, will adapt based on confirmed birds)")
        logger.info("Dual-stream detector initialized")

    def run(self, max_frames: int = 0, frame_skip: int = 5):
        """Run dual-stream detection and classification with tracking."""
        frame_count = 0
        raw_detections = 0
        classifications = 0
        logged_birds = 0

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
                    raw_detections += len(boxes)
                    logger.debug(f"Frame {frame_count}: Detected {len(boxes)} object(s)")

                    # Scale boxes to main stream coordinates
                    main_boxes = [scale_roi(box, self.sub_width, self.main_width) for box in boxes]

                    if self.classifier:
                        # Update tracker with new detections
                        finalized_birds = self.tracker.update(main_boxes, frame_count, self.classifier, main_frame)

                        classifications += len(boxes)

                        # Log finalized birds to database
                        for bird in finalized_birds:
                            logger.info(f"🐦 {bird['species'].split('(')[0].strip()}: "
                                      f"{bird['confidence']:.1%} "
                                      f"(best of {bird['num_classifications']} frames)")

                            # Register with adaptive detector for learning (scale bbox back to substream)
                            sub_bbox = scale_roi(bird['bbox'], self.main_width, self.sub_width)
                            self.detector.register_bird_detection(
                                sub_bbox,
                                bird['species'],
                                bird['confidence']
                            )

                            if self.database:
                                try:
                                    self.database.add_sighting(
                                        species=bird['species'],
                                        confidence=bird['confidence'],
                                        bbox=bird['bbox'],
                                        frame_size=(self.main_width, self.main_height),
                                        stream_source="mainstream",
                                        notes=f"Best of {bird['num_classifications']} classifications over {bird['num_frames']} frames"
                                    )
                                    logged_birds += 1
                                except Exception as db_error:
                                    logger.error(f"Database storage failed: {db_error}")

            frame_count += 1

            # Progress logging
            if frame_count % 100 == 0:
                logger.info(f"Progress: {frame_count} frames, {raw_detections} detections, "
                          f"{classifications} classifications, {logged_birds} birds logged")

        # Finalize remaining tracked birds
        if self.classifier:
            remaining_birds = self.tracker.finalize_all()
            for bird in remaining_birds:
                logger.info(f"🐦 {bird['species'].split('(')[0].strip()}: "
                          f"{bird['confidence']:.1%} "
                          f"(best of {bird['num_classifications']} frames)")

                # Register with adaptive detector for learning
                sub_bbox = scale_roi(bird['bbox'], self.main_width, self.sub_width)
                self.detector.register_bird_detection(
                    sub_bbox,
                    bird['species'],
                    bird['confidence']
                )

                if self.database:
                    try:
                        self.database.add_sighting(
                            species=bird['species'],
                            confidence=bird['confidence'],
                            bbox=bird['bbox'],
                            frame_size=(self.main_width, self.main_height),
                            stream_source="mainstream",
                            notes=f"Best of {bird['num_classifications']} classifications over {bird['num_frames']} frames"
                        )
                        logged_birds += 1
                    except Exception as db_error:
                        logger.error(f"Database storage failed: {db_error}")

        # Cleanup
        self.sub_cap.release()
        self.main_cap.release()

        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Raw detections: {raw_detections}")
        logger.info(f"Classifications: {classifications}")
        logger.info(f"Birds logged: {logged_birds} (confidence >= {self.tracker.min_confidence:.1%})")
        logger.info(f"Tracker finalized: {self.tracker.finalized_count} unique birds")

        # Log adaptive detector statistics
        stats = self.detector.get_statistics()
        if stats['bird_count'] > 0:
            logger.info(f"\n=== Adaptive Detection Learning ===")
            logger.info(f"Learned from {stats['bird_count']} confirmed bird detections")
            logger.info(f"Bird area: {stats['area_stats']['mean']:.0f} ± {stats['area_stats']['std']:.0f} px "
                       f"(range: {stats['area_stats']['min']:.0f}-{stats['area_stats']['max']:.0f})")
            logger.info(f"Bird aspect ratio: {stats['aspect_stats']['mean']:.2f} ± {stats['aspect_stats']['std']:.2f} "
                       f"(range: {stats['aspect_stats']['min']:.2f}-{stats['aspect_stats']['max']:.2f})")
            logger.info(f"Current thresholds: area={stats['current_thresholds']['min_area']}-{stats['current_thresholds']['max_area']}, "
                       f"aspect={stats['current_thresholds']['min_aspect']:.2f}-{stats['current_thresholds']['max_aspect']:.2f}")
            logger.info(f"Species seen: {', '.join(stats['species_seen'])}")


def main():
    parser = argparse.ArgumentParser(description="Dual-stream bird detection and classification with multi-frame tracking")
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
                       default=DEFAULT_LABELS_PATH,
                       help="Path to labels file")
    parser.add_argument("--edgetpu", action="store_true",
                       help="Use EdgeTPU acceleration")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                       help="Minimum confidence to log bird (0.0-1.0)")
    parser.add_argument("--tracker-max-age", type=int, default=30,
                       help="Max frames without detection before finalizing bird")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(config)

    logger.info("=== Dual-Stream Bird Detector V2 (Adaptive Learning + Multi-Frame Tracking) ===")

    # Get source URLs (RTSP or video file) based on runtime configuration
    if SOURCE_MANAGER_AVAILABLE:
        substream_url, mainstream_url = get_source_urls(config)
        logger.info(f"Source: sub={substream_url}, main={mainstream_url}")
    else:
        # Fallback to RTSP if source_manager not available
        logger.warning("source_manager not available, falling back to RTSP")
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
            db_path = Path(config.get("paths", {}).get("root", "/volume1/docker/bird-id")) / "data" / "birds.db"
            database = BirdDatabase(db_path)
            logger.info(f"Database ready: {db_path}")

    # Run detection
    try:
        detector = DualStreamDetector(
            substream_url, mainstream_url, config, classifier, database,
            min_confidence=args.min_confidence,
            tracker_max_age=args.tracker_max_age
        )
        detector.run(max_frames=args.frames, frame_skip=args.frame_skip)
    except Exception as e:
        logger.exception(f"Detection failed: {e}")
        sys.exit(1)
    finally:
        if database:
            database.close()


if __name__ == "__main__":
    main()
