#!/bin/bash
set -euo pipefail

CONFIG_PATH="${BIRDID_CONFIG:-/config/config.yaml}"
DATA_DIR="${BIRDID_DATA_DIR:-/data}"
LOG_DIR="${BIRDID_LOG_DIR:-/logs}"
HLS_DIR="${BIRDID_HLS_DIR:-/share/hls}"
CLASSIFY="${BIRDID_CLASSIFY:-1}"
FRAME_SKIP="${BIRDID_FRAME_SKIP:-5}"
MODEL_PATH="${BIRDID_MODEL_PATH:-/data/models/birds_v1.tflite}"
LABELS_PATH="${BIRDID_LABELS_PATH:-/data/models/inat_bird_labels.txt}"
USE_EDGETPU="${BIRDID_USE_EDGETPU:-0}"
STATUS_API="${BIRDID_STATUS_API:-1}"
STATUS_PORT="${BIRDID_STATUS_PORT:-8000}"

mkdir -p "${DATA_DIR}/events" "${LOG_DIR}" "${HLS_DIR}"

CLASSIFY_FLAG=()
EDGETPU_FLAG=()
if [ "${CLASSIFY}" != "0" ]; then
  CLASSIFY_FLAG=(
    --classify
    --model-path "${MODEL_PATH}"
    --labels-path "${LABELS_PATH}"
  )
  if [ "${USE_EDGETPU}" != "0" ]; then
    EDGETPU_FLAG=(--edgetpu)
  fi
fi

# Get RTSP URL from config
RTSP_URL=$(grep "main:" "${CONFIG_PATH}" | awk '{print $2}' | tr -d '"')

echo "Starting codec copy HLS (no re-encode, perfect quality)..."
ffmpeg -hide_banner -loglevel error \
  -rtsp_transport tcp \
  -i "${RTSP_URL}" \
  -c:v copy \
  -f hls \
  -hls_time 2 \
  -hls_list_size 12 \
  -hls_flags delete_segments+append_list \
  -hls_segment_filename "${HLS_DIR}/segment_%03d.ts" \
  "${HLS_DIR}/birdid.m3u8" >> "${LOG_DIR}/hls.log" 2>&1 &
HLS_PID=$!

echo "Starting detection process (outputs JSONL)..."
python3 /app/src/simple_detector.py \
  --config "${CONFIG_PATH}" \
  --log-level INFO \
  >> "${LOG_DIR}/detection.log" 2>&1 &
DETECT_PID=$!

PIDS=("${HLS_PID}" "${DETECT_PID}")

if [ "${STATUS_API}" != "0" ]; then
  echo "Starting status API on port ${STATUS_PORT}..."
  cd /app/src
  uvicorn status_api:app --host 0.0.0.0 --port "${STATUS_PORT}" >> "${LOG_DIR}/status_api.log" 2>&1 &
  STATUS_PID=$!
  PIDS+=("${STATUS_PID}")
fi

cleanup() {
  echo "Stopping Bird-ID service"
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

trap cleanup SIGINT SIGTERM
wait "${PIDS[@]}"
