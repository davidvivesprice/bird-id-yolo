#!/bin/bash
# Convert all clip videos to 360p (640x360) for testing
# Uses Synology ffmpeg7 with Intel QuickSync hardware acceleration

set -e

FFMPEG="/volume1/@appstore/ffmpeg7/bin/ffmpeg"
CLIPS_DIR="/volume1/docker/bird-id/data/clips"
OUTPUT_DIR="/volume1/docker/bird-id/data/clips_360p"

echo "Converting bird clips to 360p (640x360)..."
echo "Source: $CLIPS_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if source exists
if [ ! -d "$CLIPS_DIR" ]; then
    echo "ERROR: Clips directory not found: $CLIPS_DIR"
    exit 1
fi

# Create output directory structure
mkdir -p "$OUTPUT_DIR"

# Counter
total=0
converted=0
skipped=0

# Find all MP4 files and convert them
find "$CLIPS_DIR" -type f -name "*.mp4" | sort | while IFS= read -r video; do
    ((total++)) || true

    # Get relative path from clips dir
    rel_path="${video#$CLIPS_DIR/}"

    # Create output path
    output_file="$OUTPUT_DIR/$rel_path"
    output_dir=$(dirname "$output_file")

    # Create subdirectory if needed
    mkdir -p "$output_dir"

    # Skip if already exists
    if [ -f "$output_file" ]; then
        echo "[$total] SKIP: $(basename "$video") (already exists)"
        ((skipped++)) || true
        continue
    fi

    echo "[$total] Converting: $rel_path"

    # Convert to 640x360 using hardware acceleration (Intel QuickSync)
    if "$FFMPEG" -hwaccel vaapi \
        -vaapi_device /dev/dri/renderD128 \
        -i "$video" \
        -vf "format=nv12,hwupload,scale_vaapi=w=640:h=360" \
        -c:v h264_vaapi \
        -b:v 800k \
        -c:a copy \
        -y \
        "$output_file" 2>&1 | grep -E "Duration|frame=" | tail -5; then

        ((converted++)) || true
        echo "  ✓ Done"
    else
        echo "  ✗ Failed"
    fi
    echo ""
done

echo "========================================="
echo "Conversion complete!"
echo "Total videos found: $total"
echo "Converted: $converted"
echo "Skipped (already exist): $skipped"
echo "Output directory: $OUTPUT_DIR"
echo "========================================="
