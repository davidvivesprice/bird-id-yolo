#!/bin/bash
# Unified detection runner - supports YOLO or TFLite/EdgeTPU

set -e

# Configuration from environment variables
DETECTION_MODE="${DETECTION_MODE:-tflite}"  # yolo or tflite
RTSP_MAIN="${RTSP_MAIN:-rtsp://192.168.4.9:7447/tTHjLZrVgopARpu6}"
RTSP_SUB="${RTSP_SUB:-rtsp://192.168.4.9:7447/5CAx1qDdOe7zoLEQ}"
LOG_DIR="${LOG_DIR:-/logs-yolo}"
TEST_VIDEO_PATH="/data/clips/testing/custom_sequence.mp4"

echo "=== Bird Detection Service ==="
echo "  Mode: $DETECTION_MODE"

# Check if we should use test video instead of RTSP
if [ -f "$TEST_VIDEO_PATH" ]; then
    echo "  Source: Test video (${TEST_VIDEO_PATH})"
    USE_VIDEO=true
    VIDEO_SOURCE="$TEST_VIDEO_PATH"
else
    echo "  Source: RTSP substream"
    USE_VIDEO=false
    VIDEO_SOURCE="$RTSP_SUB"
fi

# Download Edge TPU model if needed
if [ "$DETECTION_MODE" = "tflite" ]; then
    if [ ! -f "/data/models/ssd_mobilenet_v2_edgetpu.tflite" ]; then
        echo "Downloading Edge TPU model..."
        /app/scripts/download_edgetpu_model.sh
    fi
fi

# Start appropriate detector
if [ "$DETECTION_MODE" = "tflite" ]; then
    echo "Starting TFLite Edge TPU detector..."
    if [ "$USE_VIDEO" = true ]; then
        exec python3 /app/src/tflite_detector.py \
            --video "$VIDEO_SOURCE" \
            --loop \
            --model /data/models/ssd_mobilenet_v2_edgetpu.tflite \
            --labels /data/models/coco_labels.txt \
            --output /share-yolo/detections.json \
            --log-level INFO
    else
        exec python3 /app/src/tflite_detector.py \
            --rtsp "$VIDEO_SOURCE" \
            --model /data/models/ssd_mobilenet_v2_edgetpu.tflite \
            --labels /data/models/coco_labels.txt \
            --output /share-yolo/detections.json \
            --log-level INFO
    fi
else
    echo "Starting YOLO detector..."
    if [ "$USE_VIDEO" = true ]; then
        exec python3 /app/src/yolo_detector.py \
            --video "$VIDEO_SOURCE" \
            --loop \
            --model /data/models/yolov8n.pt \
            --output /share-yolo/detections.json \
            --log-level INFO
    else
        exec python3 /app/src/yolo_detector.py \
            --rtsp "$VIDEO_SOURCE" \
            --model /data/models/yolov8n.pt \
            --output /share-yolo/detections.json \
            --log-level INFO
    fi
fi
