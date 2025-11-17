#!/usr/bin/env python3
"""
Source Manager - Handles both RTSP and video file sources for Bird-ID.
Reads runtime_config.json to determine which source to use.
"""
import json
import os
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

RUNTIME_CONFIG_PATH = Path(os.environ.get("BIRDID_DATA_DIR", "/data")) / "runtime_config.json"


def load_runtime_config() -> dict:
    """Load runtime configuration, return defaults if not found."""
    if not RUNTIME_CONFIG_PATH.exists():
        logger.info("No runtime_config.json found, using defaults (live RTSP mode)")
        return {
            "source_mode": "live",
            "video_file": None,
            "use_edgetpu": True
        }

    try:
        with open(RUNTIME_CONFIG_PATH, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded runtime config: mode={config.get('source_mode')}, "
                   f"video={config.get('video_file')}, edgetpu={config.get('use_edgetpu')}")
        return config
    except Exception as e:
        logger.error(f"Failed to load runtime config: {e}, using defaults")
        return {
            "source_mode": "live",
            "video_file": None,
            "use_edgetpu": True
        }


def get_source_urls(static_config: dict) -> Tuple[str, str]:
    """
    Get source URLs for substream and mainstream based on runtime config.

    Returns:
        (substream_url, mainstream_url)

    For RTSP: Returns separate sub and main stream URLs
    For video: Returns same video file for both (single source)
    """
    runtime_config = load_runtime_config()
    source_mode = runtime_config.get("source_mode", "live")

    if source_mode == "video":
        # Video file mode
        video_file = runtime_config.get("video_file")
        video_config = static_config.get("video", {})
        clips_dir = Path(video_config.get("clips_dir", "/data/clips_360p"))

        if not video_file:
            # Use default video from config
            video_file = video_config.get("default", "")

        if not video_file:
            logger.error("No video file specified in runtime config or static config!")
            raise ValueError("Video mode selected but no video file specified")

        # Handle absolute vs relative paths
        if video_file.startswith('/'):
            # Absolute path - use as-is
            video_path = Path(video_file)
        else:
            # Relative path - append to clips_dir
            video_path = clips_dir / video_file

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_url = str(video_path)
        logger.info(f"Using video file: {video_url}")

        # For video files, use same source for both sub and main
        return (video_url, video_url)

    else:
        # RTSP live mode (default)
        rtsp_config = static_config.get("rtsp", {})
        substream_url = rtsp_config.get("sub", "rtsp://192.168.4.9:7447/BFMOkPpZnsGP0FaW")
        mainstream_url = rtsp_config.get("main", "rtsp://192.168.4.9:7447/umtyoUf5rd0izEDr")

        logger.info(f"Using live RTSP streams")
        return (substream_url, mainstream_url)


def should_use_edgetpu() -> bool:
    """Check if EdgeTPU should be enabled based on runtime config."""
    runtime_config = load_runtime_config()
    use_edgetpu = runtime_config.get("use_edgetpu", True)

    # Also check environment variable (takes precedence)
    env_edgetpu = os.environ.get("BIRDID_USE_EDGETPU", "1") != "0"

    # Runtime config overrides environment
    result = use_edgetpu if runtime_config.get("use_edgetpu") is not None else env_edgetpu

    logger.info(f"EdgeTPU setting: {result}")
    return result
