#!/bin/bash
set -euo pipefail

CONFIG_PATH="${BIRDID_CONFIG:-/config/config.yaml}"
DATA_DIR="${BIRDID_DATA_DIR:-/data}"
LOG_DIR="${BIRDID_LOG_DIR:-/logs}"
HLS_DIR="${BIRDID_HLS_DIR:-/share/hls}"
CLASSIFY="${BIRDID_CLASSIFY:-1}"
FRAME_SKIP="${BIRDID_FRAME_SKIP:-5}"
OUTPUT_FPS="${BIRDID_OUTPUT_FPS:-10}"
HLS_PLAYLIST="${BIRDID_HLS_PLAYLIST:-birdid.m3u8}"
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

echo "Starting Bird-ID dual-stream detector (adaptive version)..."
python3 /app/src/dual_stream_detector_v2.py \
  --config "${CONFIG_PATH}" \
  "${CLASSIFY_FLAG[@]}" \
  "${EDGETPU_FLAG[@]}" \
  --frames 0 \
  --frame-skip "${FRAME_SKIP}" >> "${LOG_DIR}/dual_stream_detector.log" 2>&1 &
DETECT_PID=$!

echo "Starting annotated live preview → HLS (${OUTPUT_FPS} fps)..."
python3 /app/src/dual_stream_detector_live.py \
  --config "${CONFIG_PATH}" \
  "${CLASSIFY_FLAG[@]}" \
  "${EDGETPU_FLAG[@]}" \
  --frame-skip "${FRAME_SKIP}" \
  --output-fps "${OUTPUT_FPS}" \
  2>> "${LOG_DIR}/dual_stream_detector_live.log" \
  | ffmpeg -hide_banner -loglevel error \
      -fflags +genpts \
      -f rawvideo -pix_fmt bgr24 -s 1920x1080 -r "${OUTPUT_FPS}" -i pipe:0 \
      -c:v libx264 -preset medium \
      -b:v 6M -maxrate 8M -bufsize 12M \
      -g $((OUTPUT_FPS * 2)) -keyint_min ${OUTPUT_FPS} -sc_threshold 0 \
      -pix_fmt yuv420p \
      -f hls \
      -hls_time 2 \
      -hls_list_size 12 \
      -hls_flags delete_segments+append_list \
      -hls_segment_filename "${HLS_DIR}/segment_%03d.ts" \
      "${HLS_DIR}/${HLS_PLAYLIST}" >> "${LOG_DIR}/hls.log" 2>&1 &
HLS_PID=$!

PIDS=("${DETECT_PID}" "${HLS_PID}")

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
