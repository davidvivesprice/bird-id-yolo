# YOLO Bird Detection - Ready for Testing

## Status: ✅ Infrastructure Complete, Ready to Build and Test

**Created:** November 16, 2025
**Branch:** bird-id-yolo (isolated from TFLite system)
**Model:** YOLOv8n COCO-pretrained (generic bird detection)

---

## What's Been Set Up

### ✅ Complete Isolation
- **Separate directory**: `/volume1/docker/bird-id-yolo/`
- **Separate containers**: `birdid-yolo`, `birdid-yolo-share`
- **Separate ports**: 8001 (vs 8000)
- **Separate outputs**: `/share-yolo/` (vs `/share/`)
- **Separate logs**: `/logs-yolo/` (vs `/logs/`)
- **Separate URL**: `birdid-yolo.vivessyn.duckdns.org`
- **New git repo**: Independent version control

**Zero interference with TFLite system** - both can run simultaneously (with EdgeTPU scheduling)

### ✅ YOLO Detection Pipeline
- **Core detector**: `src/yolo_detector.py`
  - YOLOv8 object detection
  - Real-time RTSP processing
  - JSON output compatible with existing dashboard
  - Proper bounding boxes (not MOG2 crops)
  - Single-pass detection + classification

- **Docker setup**: `docker-compose-yolo.yaml`
  - birdid-yolo: Detection + HLS encoding
  - birdid-yolo-share: NGINX web server
  - CPU-only initially (no EdgeTPU conflict)

- **Service runner**: `scripts/run_yolo_service.sh`
  - FFmpeg HLS encoding (codec copy)
  - YOLO detection process
  - Status API on port 8001

### ✅ YOLO Model Downloaded
- **File**: `data/models/yolov8n.pt` (6.2 MB)
- **Type**: YOLOv8 nano - fastest variant
- **Detection**: Generic "bird" class (COCO class 14)
- **Performance**: ~12 FPS on CPU (expected)
- **Accuracy**: 37.3% mAP (sufficient for testing)

**Note**: This is generic bird detection only. For species identification (Chickadee, Titmouse, etc.), you'll need to train/download a bird species model (see below).

### ✅ Documentation
- **README-YOLO.md**: Comprehensive system documentation
  - Architecture explanation
  - Isolation strategy
  - Quick start guide
  - Model selection guide
  - Troubleshooting

- **HANDOFF.md** (this file): Ready-to-run instructions

---

## Quick Start - Test YOLO System Now

### Step 1: Build the Docker Container

```bash
cd /volume1/docker
sudo docker-compose build birdid-yolo birdid-yolo-share
```

**Expected**: 2-3 minutes to build, installs PyTorch, Ultralytics, OpenCV

### Step 2: Start the Services

```bash
sudo docker-compose up -d birdid-yolo birdid-yolo-share
```

**Expected**: Starts birdid-yolo and birdid-yolo-share containers

### Step 3: Check Logs

```bash
sudo docker logs -f birdid-yolo
```

**Expected output**:
```
Starting YOLO Bird Detection System
  RTSP Main: rtsp://192.168.4.9:7447/...
  RTSP Sub:  rtsp://192.168.4.9:7447/...
Loading YOLO model from /data/models/yolov8n.pt
Model loaded successfully
Stream opened successfully
Stream: 640x360 @ 30.0 fps
Starting detection loop...
```

### Step 4: View Detections

```bash
# Watch detection JSON
watch -n 1 'cat /volume1/docker/bird-id-yolo/share-yolo/detections.json | python3 -m json.tool'

# Or monitor logs
tail -f /volume1/docker/bird-id-yolo/logs-yolo/yolo_detection.log
```

**Remember**: Access at `https://birdid-yolo.vivessyn.duckdns.org:9444` (port 9444!)

**Expected when bird appears**:
```json
{
  "timestamp": 1731792000.123,
  "frame": 450,
  "detections": [
    {
      "x": 320,
      "y": 240,
      "w": 80,
      "h": 100,
      "label": "bird",
      "confidence": 0.78
    }
  ]
}
```

### Step 5: View Web Dashboard

Open in browser: `https://birdid-yolo.vivessyn.duckdns.org:9444`

**Expected**: Live HLS stream with green bounding boxes around detected birds

### Step 6: Stop (when done testing)

```bash
cd /volume1/docker
sudo docker-compose stop birdid-yolo birdid-yolo-share
```

---

## Compare to TFLite System

Run both systems side-by-side:

```bash
# Terminal 1: YOLO detections
tail -f /volume1/docker/bird-id-yolo/logs-yolo/yolo_detection.log | grep "Frame"

# Terminal 2: TFLite detections
tail -f /volume1/docker/bird-id/logs/detection.log | grep "Frame"
```

**What to compare**:
- ✅ Detection rate: Does YOLO detect more birds?
- ✅ False positives: Does YOLO have fewer impossible species (Blue Herons)?
- ✅ Bounding boxes: Are YOLO boxes more accurate than MOG2 crops?
- ✅ Performance: Is YOLO maintaining >10 FPS?

---

## Current Limitations & Next Steps

### ⚠️ Current Limitation: Generic Bird Detection Only

**What you'll see**: Detections labeled as "bird" (not species-specific)

**Why**: YOLOv8n COCO model only knows generic "bird" class

**Example**:
- Chickadee → Detected as "bird" ✅
- Titmouse → Detected as "bird" ✅
- Nuthatch → Detected as "bird" ✅
- Blue Heron → Will NOT detect (not present, good!) ✅

**To get species identification**, proceed to Next Steps below.

---

## Next Steps - Species Identification

### Option A: Quick Start - Roboflow Pre-trained Model (Recommended)

**Time**: 30 minutes
**Accuracy**: High (525 species)
**Effort**: Low (download + configure)

```bash
# 1. Get Roboflow API key (free): https://app.roboflow.com/

# 2. Install Roboflow SDK
pip3 install roboflow

# 3. Download bird species model
python3 << 'EOF'
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("bird-species-predictor").project("bird-species-detector")
dataset = project.version(1).download("yolov8")

# Dataset downloaded with species labels
print("Dataset ready for training!")
EOF

# 4. Train YOLOv8 on bird species
python3 << 'EOF'
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='bird-species-detector-1/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='bird_species'
)
# Trained model saved to: runs/detect/bird_species/weights/best.pt
EOF

# 5. Update docker-compose-yolo.yaml
# Change: YOLO_MODEL=/data/models/runs/detect/bird_species/weights/best.pt

# 6. Restart container
sudo docker-compose -f docker-compose-yolo.yaml restart
```

**Expected result**: Detections like "Black-capped Chickadee" (85%), "Tufted Titmouse" (92%)

### Option B: Custom Training on Your Birds (Best Accuracy)

**Time**: 2-3 hours (mostly data collection)
**Accuracy**: Best for your specific birds
**Effort**: Medium (annotation required)

```bash
# 1. Record 1-2 hours of feeder footage
# 2. Extract frames with birds (~500-1000 images)
# 3. Annotate on Roboflow (free tier):
#    - Create project
#    - Upload images
#    - Draw bounding boxes
#    - Label species
# 4. Export dataset in YOLOv8 format
# 5. Train model (same as Option A)
# 6. Deploy custom model
```

**Expected result**: >95% accuracy on YOUR specific birds

### Option C: Stay with Generic Detection

**If you just want to know "is there a bird?**" without species ID, you're done! The current setup works.

---

## Performance Optimization

### If FPS < 10:

**1. Reduce input resolution** (easiest):
```python
# In yolo_detector.py, modify detect() method:
results = self.model(frame, conf=self.confidence_threshold, imgsz=416)  # vs 640
```

**2. Skip frames** (process every 2nd frame):
```python
# In main loop:
if frame_count % 2 == 0:  # Process every 2nd frame
    detections = detector.detect(frame)
```

**3. Use ONNX export** (faster CPU inference):
```python
model.export(format='onnx')
# Use ONNX runtime for inference
```

**4. Lower confidence threshold** (fewer detections to process):
```bash
# In docker-compose-yolo.yaml:
--confidence 0.7  # vs 0.5 (only high-confidence detections)
```

### If you need EdgeTPU:

**Current**: CPU-only (no EdgeTPU conflict with TFLite system)

**To use EdgeTPU**:
1. Stop TFLite container: `sudo docker stop birdid`
2. Enable EdgeTPU in docker-compose-yolo.yaml:
   ```yaml
   devices:
     - /dev/bus/usb:/dev/bus/usb
   ```
3. Rebuild: `sudo docker-compose -f docker-compose-yolo.yaml build`
4. Restart: `sudo docker-compose -f docker-compose-yolo.yaml up -d`

**Note**: YOLO EdgeTPU support is experimental. YOLOv8 may not work. Consider YOLOv5 if EdgeTPU critical.

---

## Testing Checklist

### ✅ Basic Functionality
- [ ] Container builds successfully
- [ ] Container starts without errors
- [ ] YOLO model loads
- [ ] RTSP stream connects
- [ ] Detections appear in JSON
- [ ] Web dashboard loads
- [ ] HLS stream plays
- [ ] Bounding boxes drawn on video

### ✅ Detection Quality
- [ ] Detects birds when present
- [ ] No false positives (no Blue Herons!)
- [ ] Bounding boxes accurate (around bird, not whole frame)
- [ ] Confidence scores reasonable (>50%)
- [ ] Detection rate acceptable (catches most birds)

### ✅ Performance
- [ ] FPS >10 (check logs)
- [ ] Low latency (<2 second delay)
- [ ] No memory leaks (monitor docker stats)
- [ ] Stable for >1 hour continuous operation

### ✅ Isolation
- [ ] TFLite system still works (if running)
- [ ] YOLO system accessible at birdid-yolo.vivessyn.duckdns.org
- [ ] Separate log files
- [ ] Separate detection JSON

---

## Troubleshooting

### Container won't start

```bash
# Check logs
sudo docker logs birdid-yolo

# Common issues:
# 1. Model not found
#    → Check: ls /volume1/docker/bird-id-yolo/data/models/yolov8n.pt
# 2. RTSP stream error
#    → Test: ffplay rtsp://192.168.4.9:7447/5CAx1qDdOe7zoLEQ
# 3. Port conflict
#    → Check: sudo netstat -tulpn | grep 8001
```

### No detections appearing

```bash
# Test YOLO model directly
sudo docker exec -it birdid-yolo python3 << 'EOF'
from ultralytics import YOLO
model = YOLO('/data/models/yolov8n.pt')
results = model('https://ultralytics.com/images/bus.jpg')
print(f"Detected {len(results[0].boxes)} objects")
for box in results[0].boxes:
    print(f"  Class: {results[0].names[int(box.cls)]}, Conf: {box.conf:.2f}")
EOF

# If this works, issue is with RTSP stream or code logic
# If this fails, issue is with YOLO installation
```

### Low FPS

```bash
# Check actual FPS in logs
grep "frames processed" /volume1/docker/bird-id-yolo/logs-yolo/yolo_detection.log

# If <10 FPS, try:
# 1. Reduce resolution (imgsz=416)
# 2. Skip frames (process every 2nd)
# 3. Lower confidence (0.7 vs 0.5)
```

### Web dashboard not loading

```bash
# Check NGINX container
sudo docker logs birdid-yolo-share

# Check Traefik routing
sudo docker logs traefik | grep birdid-yolo

# Verify DNS
ping birdid-yolo.vivessyn.duckdns.org
```

---

## Files Created

```
/volume1/docker/bird-id-yolo/
├── README-YOLO.md              # Comprehensive documentation
├── HANDOFF.md                  # This file (quick start)
├── docker-compose-yolo.yaml    # Container orchestration
├── Dockerfile-yolo             # YOLO container definition
├── src/
│   └── yolo_detector.py        # Main YOLO detection script (289 lines)
├── scripts/
│   └── run_yolo_service.sh     # Container entrypoint
├── data/
│   └── models/
│       └── yolov8n.pt          # YOLOv8 nano model (6.2 MB)
├── share-yolo/
│   ├── index.html              # Web dashboard
│   ├── nginx.conf              # NGINX config
│   └── detections.json         # Real-time output (created on run)
└── logs-yolo/                  # Log directory (created on run)
```

---

## Git Commits

```
762f9c3 - Initial YOLO branch - copied from bird-id TFLite system
9ba582d - Add YOLO detection infrastructure
(yolov8n.pt not committed - excluded by .gitignore)
```

---

## Success Criteria Review

### ✅ Completed
- [x] Complete isolation (separate directory, containers, ports, files)
- [x] YOLO detection pipeline implemented
- [x] Docker configuration created
- [x] YOLO model downloaded (yolov8n.pt)
- [x] Comprehensive documentation
- [x] Web interface configured
- [x] Service runner script
- [x] Git repository initialized

### ⏳ Pending (Your Tasks)
- [ ] Build Docker container
- [ ] Start services
- [ ] Test on live stream
- [ ] Compare to TFLite system
- [ ] Download/train bird species model (if desired)
- [ ] Performance optimization (if needed)

---

## Commands Summary

```bash
# Build
cd /volume1/docker
sudo docker-compose build birdid-yolo birdid-yolo-share

# Start
sudo docker-compose up -d birdid-yolo birdid-yolo-share

# Monitor
sudo docker logs -f birdid-yolo
tail -f /volume1/docker/bird-id-yolo/logs-yolo/yolo_detection.log

# Stop
sudo docker-compose stop birdid-yolo birdid-yolo-share

# View detections
watch -n 1 'cat /volume1/docker/bird-id-yolo/share-yolo/detections.json | python3 -m json.tool'

# Web dashboard (remember port 9444!)
open https://birdid-yolo.vivessyn.duckdns.org:9444
```

---

## Final Notes

**This system is ready to test immediately with generic bird detection.**

For species identification (Chickadee, Titmouse, etc.), follow "Next Steps - Species Identification" above.

Both YOLO and TFLite systems can run simultaneously for A/B comparison. They are completely isolated.

**Good luck!** 🐦

---

*Handoff completed: November 16, 2025*
*YOLO Branch Claude*
