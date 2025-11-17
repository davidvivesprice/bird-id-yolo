#!/usr/bin/env python3
"""Adaptive motion detector that learns bird sizes over time."""
import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveBirdDetector:
    """Motion detector that adapts to your specific feeder setup."""

    def __init__(self, width: int, height: int,
                 mask: Optional[List[Tuple[int, int]]] = None,
                 initial_min_area: int = 50000,  # Conservative start
                 initial_max_area: int = 500000,  # Max reasonable bird size
                 min_aspect: float = 0.3,  # Filter out super wide streaks
                 max_aspect: float = 3.0,  # Filter out super tall streaks
                 learning_window: int = 100):  # Samples for adaptation

        self.width = width
        self.height = height
        self.mask = mask

        # Adaptive thresholds
        self.min_area = initial_min_area
        self.max_area = initial_max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

        # Learning history
        self.learning_window = learning_window
        self.detection_sizes = deque(maxlen=learning_window)
        self.bird_detections = []  # Confirmed birds (from classifier)

        # MOG2 background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

        # Create mask if provided
        self.roi_mask = None
        if mask:
            self.roi_mask = np.zeros((height, width), dtype=np.uint8)
            pts = np.array(mask, dtype=np.int32)
            cv2.fillPoly(self.roi_mask, [pts], 255)

        logger.info(f"Adaptive detector initialized: min_area={self.min_area}, "
                   f"max_area={self.max_area}, aspect={self.min_aspect}-{self.max_aspect}")

    def update(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect motion and return filtered bounding boxes."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Apply ROI mask if provided
        if self.roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, self.roi_mask)

        # Remove shadows (value 127 in MOG2 output)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter and extract bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / h if h > 0 else 0

            # Apply smart filters
            if self._is_valid_detection(area, aspect, w, h):
                boxes.append((x, y, w, h))
                # Track for learning
                self.detection_sizes.append({
                    'area': area,
                    'aspect': aspect,
                    'width': w,
                    'height': h
                })

        return boxes

    def _is_valid_detection(self, area: int, aspect: float,
                           width: int, height: int) -> bool:
        """Determine if detection passes all filters."""
        # Size filter
        if area < self.min_area or area > self.max_area:
            return False

        # Aspect ratio filter (reject thin streaks)
        if aspect < self.min_aspect or aspect > self.max_aspect:
            return False

        # Minimum dimensions (reject tiny detections)
        if width < 50 or height < 50:
            return False

        return True

    def register_bird_detection(self, bbox: Tuple[int, int, int, int],
                               species: str, confidence: float):
        """Register a confirmed bird detection for learning."""
        x, y, w, h = bbox
        area = w * h
        aspect = w / h if h > 0 else 0

        # Only learn from high-confidence detections
        if confidence >= 0.7 and species != 'background':
            self.bird_detections.append({
                'area': area,
                'aspect': aspect,
                'width': w,
                'height': h,
                'species': species,
                'confidence': confidence
            })

            logger.debug(f"Learned bird: {species} {w}x{h}={area}px, aspect={aspect:.2f}")

            # Adapt thresholds if we have enough data
            if len(self.bird_detections) >= 10:
                self._adapt_thresholds()

    def _adapt_thresholds(self):
        """Automatically adjust thresholds based on confirmed bird detections."""
        if len(self.bird_detections) < 10:
            return

        areas = [d['area'] for d in self.bird_detections]
        aspects = [d['aspect'] for d in self.bird_detections]

        # Calculate statistics
        area_mean = np.mean(areas)
        area_std = np.std(areas)
        aspect_mean = np.mean(aspects)
        aspect_std = np.std(aspects)

        # Set adaptive thresholds (mean ± 2 std dev)
        new_min_area = max(10000, int(area_mean - 2 * area_std))
        new_max_area = min(1000000, int(area_mean + 2 * area_std))
        new_min_aspect = max(0.2, aspect_mean - 2 * aspect_std)
        new_max_aspect = min(5.0, aspect_mean + 2 * aspect_std)

        # Only update if significantly different
        if abs(new_min_area - self.min_area) / self.min_area > 0.2:
            logger.info(f"Adapted min_area: {self.min_area} → {new_min_area}")
            self.min_area = new_min_area

        if abs(new_max_area - self.max_area) / self.max_area > 0.2:
            logger.info(f"Adapted max_area: {self.max_area} → {new_max_area}")
            self.max_area = new_max_area

        if abs(new_min_aspect - self.min_aspect) > 0.1:
            logger.info(f"Adapted min_aspect: {self.min_aspect:.2f} → {new_min_aspect:.2f}")
            self.min_aspect = new_min_aspect

        if abs(new_max_aspect - self.max_aspect) > 0.1:
            logger.info(f"Adapted max_aspect: {self.max_aspect:.2f} → {new_max_aspect:.2f}")
            self.max_aspect = new_max_aspect

    def get_statistics(self) -> dict:
        """Get current detection statistics."""
        if not self.bird_detections:
            return {
                'bird_count': 0,
                'current_thresholds': {
                    'min_area': self.min_area,
                    'max_area': self.max_area,
                    'min_aspect': self.min_aspect,
                    'max_aspect': self.max_aspect
                }
            }

        areas = [d['area'] for d in self.bird_detections]
        aspects = [d['aspect'] for d in self.bird_detections]

        return {
            'bird_count': len(self.bird_detections),
            'area_stats': {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas)
            },
            'aspect_stats': {
                'mean': np.mean(aspects),
                'std': np.std(aspects),
                'min': np.min(aspects),
                'max': np.max(aspects)
            },
            'current_thresholds': {
                'min_area': self.min_area,
                'max_area': self.max_area,
                'min_aspect': self.min_aspect,
                'max_aspect': self.max_aspect
            },
            'species_seen': list(set(d['species'] for d in self.bird_detections))
        }
