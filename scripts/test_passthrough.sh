#\!/bin/bash
# Minimal passthrough test - no detection, no annotation
python3 -c "
import cv2
import sys

cap = cv2.VideoCapture('rtsp://192.168.4.9:7447/tTHjLZrVgopARpu6', cv2.CAP_FFMPEG)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    sys.stdout.buffer.write(frame.tobytes())
    sys.stdout.buffer.flush()
" 2>/dev/null | ffmpeg -hide_banner -loglevel error \
    -f rawvideo -pix_fmt bgr24 -s 1920x1080 -r 25 -i pipe:0 \
    -c:v libx264 -preset medium -b:v 6M -maxrate 8M -bufsize 12M \
    -g 50 -keyint_min 25 -sc_threshold 0 -pix_fmt yuv420p \
    -f hls -hls_time 2 -hls_list_size 12 \
    -hls_flags delete_segments+append_list \
    -hls_segment_filename /volume1/docker/bird-id/share/hls/test_%03d.ts \
    /volume1/docker/bird-id/share/hls/test.m3u8
