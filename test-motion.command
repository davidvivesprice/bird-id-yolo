#!/bin/bash
# Bird-ID Motion Detection Tester
#
# Usage:
#   1. Double-click: Processes all MP4s in data/clips/inbox/
#   2. Drag & Drop: Drag MP4 file(s) onto this script

cd "$(dirname "$0")"

echo "================================================"
echo "  Bird-ID Motion Detection Tester"
echo "================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Create inbox/outbox if they don't exist
mkdir -p data/clips/inbox
mkdir -p data/clips/output

# Check if files were dragged onto the script
if [ $# -gt 0 ]; then
    # Files were dragged - process them
    echo "Processing dragged files..."
    for video in "$@"; do
        if [[ "$video" == *.mp4 ]] || [[ "$video" == *.MP4 ]]; then
            filename=$(basename "$video" .mp4)
            filename=$(basename "$filename" .MP4)
            output_dir="data/clips/output/${filename}_detected"

            echo ""
            echo "📹 Processing: $filename"
            echo "   Output: $output_dir"

            python src/motion_detector.py \
                --source "$video" \
                --output "$output_dir" \
                --mask \
                --frames 0

            # Create video from frames
            if [ -d "$output_dir" ] && [ "$(ls -A $output_dir/*.jpg 2>/dev/null)" ]; then
                echo "   Creating video..."
                ffmpeg -y -framerate 30 -pattern_type glob -i "${output_dir}/*.jpg" \
                    -c:v libx264 -pix_fmt yuv420p \
                    "data/clips/output/${filename}_detected.mp4" \
                    -loglevel error

                if [ $? -eq 0 ]; then
                    echo "   ✅ Done: data/clips/output/${filename}_detected.mp4"
                    # Clean up frames
                    rm -rf "$output_dir"
                fi
            fi
        else
            echo "⚠️  Skipping (not MP4): $video"
        fi
    done
else
    # No files dragged - process inbox folder
    echo "No files dragged. Checking inbox folder..."
    echo "Inbox: data/clips/inbox/"
    echo ""

    shopt -s nullglob
    videos=(data/clips/inbox/*.mp4 data/clips/inbox/*.MP4)

    if [ ${#videos[@]} -eq 0 ]; then
        echo "📁 Inbox is empty!"
        echo ""
        echo "Drop MP4 files in: data/clips/inbox/"
        echo "or drag MP4 files onto this script."
    else
        for video in "${videos[@]}"; do
            filename=$(basename "$video" .mp4)
            filename=$(basename "$filename" .MP4)
            output_dir="data/clips/output/${filename}_detected"

            echo "📹 Processing: $filename"
            echo "   Output: $output_dir"

            python src/motion_detector.py \
                --source "$video" \
                --output "$output_dir" \
                --mask \
                --frames 0

            # Create video from frames
            if [ -d "$output_dir" ] && [ "$(ls -A $output_dir/*.jpg 2>/dev/null)" ]; then
                echo "   Creating video..."
                ffmpeg -y -framerate 30 -pattern_type glob -i "${output_dir}/*.jpg" \
                    -c:v libx264 -pix_fmt yuv420p \
                    "data/clips/output/${filename}_detected.mp4" \
                    -loglevel error

                if [ $? -eq 0 ]; then
                    echo "   ✅ Done: data/clips/output/${filename}_detected.mp4"
                    # Move processed file
                    mv "$video" "data/clips/output/${filename}_original.mp4"
                    # Clean up frames
                    rm -rf "$output_dir"
                fi
            fi
        done
    fi
fi

echo ""
echo "================================================"
echo "All done! Check data/clips/output/"
echo "================================================"
echo ""
echo "Press any key to close..."
read -n 1
