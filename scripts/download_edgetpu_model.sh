#!/bin/bash
# Download pre-compiled Edge TPU models

MODEL_DIR="/data/models"
mkdir -p "$MODEL_DIR"

# Download MobileNet SSD v2 (COCO) - Edge TPU compiled
echo "Downloading MobileNet SSD v2 Edge TPU model..."
cd "$MODEL_DIR"

# Download model using curl
curl -L -o ssd_mobilenet_v2_edgetpu.tflite \
    https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite

# Download labels using curl
curl -L -o coco_labels.txt \
    https://github.com/google-coral/test_data/raw/master/coco_labels.txt

echo "Edge TPU model downloaded successfully!"
ls -lh "$MODEL_DIR"/ssd_mobilenet* "$MODEL_DIR"/coco_labels.txt
