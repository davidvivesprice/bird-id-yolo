#!/bin/bash
# Bird Detection Service Runner
# Supports YOLO or TFLite/EdgeTPU detection modes
# Runs HLS encoding + Detection + Status API

set -e

# Configuration from environment variables
DETECTION_MODE="${DETECTION_MODE:-tflite}"  # yolo or tflite
RTSP_MAIN="${RTSP_MAIN:-rtsp://192.168.4.9:7447/tTHjLZrVgopARpu6}"
RTSP_SUB="${RTSP_SUB:-rtsp://192.168.4.9:7447/5CAx1qDdOe7zoLEQ}"
HLS_DIR="${HLS_DIR:-/share-yolo/hls}"
STATUS_PORT="${STATUS_PORT:-8001}"
LOG_DIR="${LOG_DIR:-/logs-yolo}"
YOLO_MODEL="${YOLO_MODEL:-/data/models/yolov8n.pt}"
TFLITE_MODEL_TPU="${TFLITE_MODEL_TPU:-/data/models/ssd_mobilenet_v2_edgetpu.tflite}"
TFLITE_MODEL_CPU="${TFLITE_MODEL_CPU:-/data/models/ssd_mobilenet_v2_coco.tflite}"
# Use Edge TPU model (installed from Google's repo with proper versions)
TFLITE_MODEL="$TFLITE_MODEL_TPU"
TFLITE_LABELS="${TFLITE_LABELS:-/data/models/coco_labels.txt}"
DETECTIONS_FILE="${DETECTIONS_FILE:-/share-yolo/detections.json}"
TEST_VIDEO_PATH="/data/clips/testing/custom_sequence.mp4"

# Check if we should use test video instead of RTSP
if [ -f "$TEST_VIDEO_PATH" ]; then
    echo "Found test video at $TEST_VIDEO_PATH - using video mode"
    USE_VIDEO=true
    VIDEO_SOURCE="$TEST_VIDEO_PATH"
else
    echo "No test video found - using RTSP mode"
    USE_VIDEO=false
    VIDEO_SOURCE="$RTSP_MAIN"
fi

# Cleanup function
cleanup() {
    echo "Stopping Bird-ID service..."
    pkill -P $$ || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Clear old HLS segments
rm -rf "$HLS_DIR"/*

echo "Starting Bird Detection System"
echo "  Detection Mode: $DETECTION_MODE"
echo "  RTSP Main: $RTSP_MAIN"
echo "  RTSP Sub:  $RTSP_SUB"
echo "  HLS Dir:   $HLS_DIR"
echo "  Status API: Port $STATUS_PORT"

# Download Edge TPU model if needed
if [ "$DETECTION_MODE" = "tflite" ]; then
    if [ ! -f "$TFLITE_MODEL" ]; then
        echo "Downloading Edge TPU model..."
        /app/scripts/download_edgetpu_model.sh
    fi
fi

# Start HLS encoding
if [ "$USE_VIDEO" = true ]; then
    # Video file mode with looping - codec copy for low CPU usage
    echo "Starting HLS with codec copy (video file mode)..."
    ffmpeg \
        -stream_loop -1 \
        -i "$VIDEO_SOURCE" \
        -c:v copy \
        -c:a aac \
        -f hls \
        -hls_time 2 \
        -hls_list_size 5 \
        -hls_flags delete_segments+append_list \
        -hls_segment_filename "$HLS_DIR/segment_%03d.ts" \
        "$HLS_DIR/stream.m3u8" \
        > "$LOG_DIR/hls.log" 2>&1 &
else
    # RTSP mode - codec copy for live streams
    echo "Starting HLS with codec copy (RTSP mode)..."
    ffmpeg \
        -rtsp_transport tcp \
        -i "$VIDEO_SOURCE" \
        -c:v copy \
        -f hls \
        -hls_time 2 \
        -hls_list_size 5 \
        -hls_flags delete_segments+append_list \
        -hls_segment_filename "$HLS_DIR/segment_%03d.ts" \
        "$HLS_DIR/stream.m3u8" \
        > "$LOG_DIR/hls.log" 2>&1 &
fi

# Start appropriate detection process
if [ "$DETECTION_MODE" = "tflite" ]; then
    echo "Starting TFLite Edge TPU detection process..."
    if [ "$USE_VIDEO" = true ]; then
        python3 /app/src/tflite_detector.py \
            --video "$VIDEO_SOURCE" \
            --loop \
            --model "$TFLITE_MODEL" \
            --labels "$TFLITE_LABELS" \
            --output "$DETECTIONS_FILE" \
            --log-level INFO \
            > "$LOG_DIR/tflite_detection.log" 2>&1 &
    else
        python3 /app/src/tflite_detector.py \
            --rtsp "$RTSP_SUB" \
            --model "$TFLITE_MODEL" \
            --labels "$TFLITE_LABELS" \
            --output "$DETECTIONS_FILE" \
            --log-level INFO \
            > "$LOG_DIR/tflite_detection.log" 2>&1 &
    fi
else
    echo "Starting YOLO detection process..."
    if [ "$USE_VIDEO" = true ]; then
        python3 /app/src/yolo_detector.py \
            --video "$VIDEO_SOURCE" \
            --loop \
            --model "$YOLO_MODEL" \
            --output "$DETECTIONS_FILE" \
            --log-level INFO \
            > "$LOG_DIR/yolo_detection.log" 2>&1 &
    else
        python3 /app/src/yolo_detector.py \
            --rtsp "$RTSP_SUB" \
            --model "$YOLO_MODEL" \
            --output "$DETECTIONS_FILE" \
            --log-level INFO \
            > "$LOG_DIR/yolo_detection.log" 2>&1 &
    fi
fi

# Start status API
echo "Starting status API on port $STATUS_PORT..."
python3 /app/src/status_api.py \
    --port "$STATUS_PORT" \
    > "$LOG_DIR/status_api.log" 2>&1 &

# Wait for all background processes
wait
