# Bird-ID YOLO - Real-time Object Detection Branch

This is the **YOLO-based real-time bird detection** branch, running completely isolated from the TFLite motion-triggered system.

## Architecture

### YOLO Approach (This Branch)
```
RTSP Stream → YOLO Detection → Bounding Boxes + Species → Draw Overlay → HLS + JSON
```

**Single-pass processing:**
- YOLO performs detection AND classification in one inference
- No motion detection required
- Proper bounding boxes from object detector
- Real-time overlay on HLS stream

### vs TFLite Approach (Other Branch)
```
RTSP Stream → MOG2 Motion Detection → Crop → TFLite Classify → Draw Overlay → HLS + JSON
```

**Two-stage processing:**
- MOG2 finds motion (unreliable bounding boxes)
- TFLite classifies crops (can misclassify due to poor crops)
- Real-time overlay

## Isolation Strategy

### Separate Resources
- **Directory**: `/volume1/docker/bird-id-yolo/` (completely separate)
- **Containers**: `birdid-yolo`, `birdid-yolo-share` (different names)
- **Ports**: Status API on 8001 (vs 8000)
- **Output**: `/share-yolo/detections.json` (vs `/share/detections.json`)
- **HLS**: `/share-yolo/hls/` (vs `/share/hls/`)
- **Logs**: `/logs-yolo/` (vs `/logs/`)
- **URL**: `https://birdid-yolo.vivessyn.duckdns.org` (vs `birdid.vivessyn.duckdns.org`)

### Shared Resources
- **RTSP Streams**: Same streams (multiple readers OK)
- **Docker Network**: `internal` network (containers can coexist)

### EdgeTPU Strategy
- **Phase 1**: YOLO on CPU only (TFLite system keeps EdgeTPU)
- **Phase 2**: Stop TFLite, test YOLO with EdgeTPU (if compatible)
- **Phase 3**: Compare performance, choose winner

## Quick Start

### Build and Run

```bash
cd /volume1/docker/bird-id-yolo

# Build the container
sudo docker-compose -f docker-compose-yolo.yaml build

# Start the services
sudo docker-compose -f docker-compose-yolo.yaml up -d

# View logs
sudo docker logs -f birdid-yolo

# View web interface
open https://birdid-yolo.vivessyn.duckdns.org
```

### Stop

```bash
sudo docker-compose -f docker-compose-yolo.yaml down
```

## YOLO Model Setup

### Option 1: Use Pre-trained YOLOv8 (Quick Start)

```bash
# Download YOLOv8 nano model (fastest)
cd /volume1/docker/bird-id-yolo/data/models/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Or medium model (more accurate)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

Update `docker-compose-yolo.yaml`:
```yaml
environment:
  - YOLO_MODEL=/data/models/yolov8n.pt  # or yolov8m.pt
```

**Note**: Generic YOLO models detect "bird" class but won't identify species. Good for initial testing.

### Option 2: Use Bird-Specific YOLO Model (Recommended)

Search for pre-trained bird detection models:
- **Roboflow Universe**: https://universe.roboflow.com/ (search "bird detection")
- **GitHub**: Search "yolo bird detection"
- **Ultralytics Hub**: https://hub.ultralytics.com/

Example model formats:
- `.pt` files (PyTorch) - preferred
- `.onnx` files - also supported

### Option 3: Train Custom Model (Advanced)

```bash
# Collect bird images from your feeder
# Annotate with bounding boxes (use Roboflow or Label Studio)
# Train YOLO on your specific bird species

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='birds.yaml', epochs=100, imgsz=640)
```

## Testing

### Test on Video File

```bash
# Copy test video
cp /volume1/docker/bird-id/testing/various_birds.mp4 /volume1/docker/bird-id-yolo/data/

# Modify yolo_detector.py to accept video file
# Or use Ultralytics CLI directly:
yolo detect predict model=/data/models/yolov8n.pt source=/data/various_birds.mp4
```

### Monitor Detections

```bash
# Watch detection JSON
watch -n 1 'cat /volume1/docker/bird-id-yolo/share-yolo/detections.json | python3 -m json.tool'

# Watch logs
tail -f /volume1/docker/bird-id-yolo/logs-yolo/yolo_detection.log
```

### Compare to TFLite System

Run both systems simultaneously and compare:

```bash
# Terminal 1: YOLO detections
tail -f /volume1/docker/bird-id-yolo/logs-yolo/yolo_detection.log | grep "Frame"

# Terminal 2: TFLite detections
tail -f /volume1/docker/bird-id/logs/detection.log | grep "Frame"

# Compare results
```

## Performance Benchmarking

### Expected Performance

**CPU-only (no EdgeTPU):**
- YOLOv8n (nano): ~50-100ms per frame (~10-20 fps)
- YOLOv8m (medium): ~100-200ms per frame (~5-10 fps)

**With EdgeTPU (if compatible):**
- Target: <50ms per frame (>20 fps)

### Measure FPS

```bash
# Add FPS logging to yolo_detector.py
# Check logs for processing time
grep "fps" /volume1/docker/bird-id-yolo/logs-yolo/yolo_detection.log
```

## Success Criteria

### Must Have
- [x] Isolated from TFLite system (separate containers, files, ports)
- [ ] YOLO model loaded and running
- [ ] Detections appearing in JSON output
- [ ] Accurate species detection (no impossible birds!)
- [ ] Real-time performance (>10 fps on CPU)

### Nice to Have
- [ ] Better accuracy than TFLite system
- [ ] EdgeTPU acceleration working
- [ ] Species identification (not just "bird")
- [ ] Tracking across frames

## Troubleshooting

### Container Won't Start

```bash
# Check logs
sudo docker logs birdid-yolo

# Common issues:
# - YOLO model not found: Check path in docker-compose-yolo.yaml
# - RTSP stream error: Test with ffplay rtsp://...
# - Python import errors: Rebuild container
```

### No Detections

```bash
# Test YOLO model directly
docker exec -it birdid-yolo python3 << 'EOF'
from ultralytics import YOLO
model = YOLO('/data/models/yolov8n.pt')
results = model('https://ultralytics.com/images/bus.jpg')
print(results)
EOF

# If this works, issue is with RTSP stream or code logic
```

### Low FPS

- Try smaller model (yolov8n vs yolov8m)
- Reduce input resolution
- Skip frames (process every Nth frame)
- Enable EdgeTPU acceleration

## Development

### Modify Detection Logic

Edit `/volume1/docker/bird-id-yolo/src/yolo_detector.py` and rebuild:

```bash
sudo docker-compose -f docker-compose-yolo.yaml build
sudo docker-compose -f docker-compose-yolo.yaml restart
```

### Add Features

- **Tracking**: Use YOLO's built-in tracking (`model.track()`)
- **Regions of Interest**: Crop frame before YOLO inference
- **Multi-camera**: Add additional RTSP streams
- **Database**: Log detections to SQLite/PostgreSQL

## File Structure

```
bird-id-yolo/
├── docker-compose-yolo.yaml   # Container orchestration
├── Dockerfile-yolo             # YOLO dependencies (torch, ultralytics)
├── README-YOLO.md             # This file
├── src/
│   └── yolo_detector.py       # Main YOLO detection script
├── scripts/
│   └── run_yolo_service.sh    # Container entrypoint
├── data/
│   └── models/
│       └── yolov8n.pt         # YOLO model (download separately)
├── share-yolo/
│   ├── detections.json        # Real-time detection output
│   ├── hls/                   # HLS video segments
│   ├── index.html             # Web dashboard
│   └── nginx.conf             # NGINX config
└── logs-yolo/
    ├── yolo_detection.log     # Detection logs
    ├── hls.log                # HLS encoding logs
    └── status_api.log         # API logs
```

## Next Steps

1. **Download YOLO model** (see Option 1 above)
2. **Build container**: `sudo docker-compose -f docker-compose-yolo.yaml build`
3. **Start services**: `sudo docker-compose -f docker-compose-yolo.yaml up -d`
4. **Test on live stream**: Check `https://birdid-yolo.vivessyn.duckdns.org`
5. **Compare to TFLite**: Run both systems and evaluate accuracy
6. **Optimize**: Fine-tune model, try EdgeTPU, improve FPS

## Resources

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **YOLOv8 Models**: https://github.com/ultralytics/ultralytics
- **Bird Detection Datasets**: https://universe.roboflow.com/
- **YOLO Training Tutorial**: https://docs.ultralytics.com/modes/train/

---

**Created**: Nov 16, 2025
**Branch**: YOLO Real-time Detection
**Parent**: bird-id TFLite system
**Status**: Initial setup complete, ready for model selection and testing
