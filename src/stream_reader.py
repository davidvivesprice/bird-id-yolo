#!/usr/bin/env python3
"""Lightweight RTSP frame capture for the bird-id project."""
import argparse
import os
from pathlib import Path
import cv2

DEFAULT_RTSP = "rtsp://192.168.4.9:7447/tTHjLZrVgopARpu6"
DEFAULT_OUT = Path("/volume1/docker/bird-id/data/test/frame.jpg")


def capture_frame(rtsp_url: str, output_path: Path) -> None:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from RTSP stream")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write snapshot to {output_path}")
    print(f"Saved snapshot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grab frames from the bird RTSP feed.")
    parser.add_argument("--rtsp", default=os.getenv("BIRD_RTSP", DEFAULT_RTSP), help="RTSP URL")
    parser.add_argument("--snapshot", action="store_true", help="Capture a single frame and exit")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Where to save the snapshot when --snapshot is used")
    args = parser.parse_args()

    if args.snapshot:
        capture_frame(args.rtsp, Path(args.output))
    else:
        parser.error("Continuous capture loop not implemented yet. Use --snapshot for now.")


if __name__ == "__main__":
    main()
