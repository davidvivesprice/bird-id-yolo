#!/bin/bash
# Helper to compile the MobileNetV2 bird model for EdgeTPU using Google's container.
set -euo pipefail

MODEL_DIR="${1:-/volume1/docker/bird-id/data/models}"
MODEL_NAME="${2:-birds_v1.tflite}"
OUTPUT_DIR="${3:-$MODEL_DIR}"

if [ ! -f "${MODEL_DIR}/${MODEL_NAME}" ]; then
  echo "Model ${MODEL_DIR}/${MODEL_NAME} not found" >&2
  exit 1
fi

echo "Running EdgeTPU compiler in Docker..."
docker run --rm \
  -v "${MODEL_DIR}":/models \
  gcr.io/coral-project/edgetpu-compiler \
  edgetpu_compiler "/models/${MODEL_NAME}" -o /models/

echo "Compiled model written to ${OUTPUT_DIR}"
