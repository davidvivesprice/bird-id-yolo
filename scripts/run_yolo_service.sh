#!/bin/bash
# YOLO Bird Detection Service Runner
# Runs HLS encoding + YOLO detection + Status API

set -e

# Configuration from environment variables
RTSP_MAIN="${RTSP_MAIN:-rtsp://192.168.4.9:7447/tTHjLZrVgopARpu6}"
RTSP_SUB="${RTSP_SUB:-rtsp://192.168.4.9:7447/5CAx1qDdOe7zoLEQ}"
HLS_DIR="${HLS_DIR:-/share-yolo/hls}"
STATUS_PORT="${STATUS_PORT:-8001}"
LOG_DIR="${LOG_DIR:-/logs-yolo}"

# Cleanup function
cleanup() {
    echo "Stopping YOLO Bird-ID service..."
    pkill -P $$ || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Clear old HLS segments
rm -rf "$HLS_DIR"/*

echo "Starting YOLO Bird Detection System"
echo "  RTSP Main: $RTSP_MAIN"
echo "  RTSP Sub:  $RTSP_SUB"
echo "  HLS Dir:   $HLS_DIR"
echo "  Status API: Port $STATUS_PORT"

# Start HLS encoding (codec copy - no re-encode)
echo "Starting codec copy HLS (no re-encode, perfect quality)..."
ffmpeg \
    -rtsp_transport tcp \
    -i "$RTSP_MAIN" \
    -c:v copy \
    -f hls \
    -hls_time 2 \
    -hls_list_size 5 \
    -hls_flags delete_segments+append_list \
    -hls_segment_filename "$HLS_DIR/segment_%03d.ts" \
    "$HLS_DIR/stream.m3u8" \
    > "$LOG_DIR/hls.log" 2>&1 &

# Start YOLO detection process
echo "Starting YOLO detection process..."
python3 /app/src/yolo_detector.py \
    --rtsp "$RTSP_SUB" \
    --model "$YOLO_MODEL" \
    --output "$DETECTIONS_FILE" \
    --log-level INFO \
    > "$LOG_DIR/yolo_detection.log" 2>&1 &

# Start status API
echo "Starting status API on port $STATUS_PORT..."
python3 /app/src/status_api.py \
    --port "$STATUS_PORT" \
    > "$LOG_DIR/status_api.log" 2>&1 &

# Wait for all background processes
wait
