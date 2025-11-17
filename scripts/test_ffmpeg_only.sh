#!/bin/bash
# Minimal test - ffmpeg only, no Python/OpenCV
# If this has ghosting: ffmpeg re-encode is the problem
# If this is clear: OpenCV is adding the ghosting

RTSP_URL="rtsp://192.168.4.9:7447/tTHjLZrVgopARpu6"
HLS_DIR="/share/hls"

ffmpeg -hide_banner -loglevel error \
    -rtsp_transport tcp \
    -i "${RTSP_URL}" \
    -c:v libx264 -preset medium \
    -b:v 6M -maxrate 8M -bufsize 12M \
    -g 50 -keyint_min 25 -sc_threshold 0 \
    -pix_fmt yuv420p \
    -f hls -hls_time 2 -hls_list_size 12 \
    -hls_flags delete_segments+append_list \
    -hls_segment_filename "${HLS_DIR}/test_%03d.ts" \
    "${HLS_DIR}/test.m3u8"
