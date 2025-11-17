# Implementation Log - Config & Logging System

## Date: 2025-11-15

### 1. Docker Service + Traefik Live View
- Added `scripts/run_service.sh` + Docker entrypoint to run `dual_stream_detector.py` and the annotated preview in one container.
- HLS segments land in `share/hls`; `birdid-share` (nginx) serves them via Traefik at `https://birdid.vivessyn.duckdns.org`.
- New compose services: `birdid` (builds from repo) and `birdid-share`.
- `dual_stream_detector_live.py` now emits JPEG frames so ffmpeg can ingest them from stdin.
- Added FastAPI status API (`/api/status`, `/api/recent`) proxied through the same Traefik host for future dashboard work.
- Built `/api/species` endpoint on top of `BirdDatabase` and a client-side dashboard in `share/index.html` that fetches status/recent/species data.
- Added `scripts/compile_edgetpu.sh` helper so compiling the TFLite model via Google’s container is a single command.

### 2. Docs & README
- Documented Docker workflow, service URLs, and architecture updates in `README.md`, `docs/PROJECT_OVERVIEW.md`, and `docs/ROADMAP.md`.
- Captured inspiration + future work in `docs/FUTURE_VISION.md`.

### 3. Next Steps
- EdgeTPU model compilation + classifier auto-detection.
- Live dashboard/API inside the container for stats + recent detections.
- Calibration + personalization phase once the service is stable.

---

## Date: 2025-11-14

## What Was Implemented

### 1. Centralized Configuration (config.yaml)
Created `/volume1/docker/bird-id/config.yaml` with:
- **RTSP URLs**: Main and sub streams from UniFi camera
- **Motion Detection Parameters**: 
  - ROI coordinates for bird feeder area
  - Threshold settings (diff_threshold, min_area, history)
  - Consecutive frame filtering
- **File Paths**: All project paths in one place
- **Logging Configuration**: Level, rotation, output settings
- **Capture Settings**: JPEG quality, snapshot locations

**Benefits:**
- No more hardcoded values scattered in code
- Easy tuning without touching Python
- Consistent paths across all scripts

### 2. Structured Logging System
Enhanced `src/motion_detector.py` with:
- Rotating log files (10MB max, 5 backups)
- Timestamped entries with severity levels
- Console + file output
- Progress tracking (every 50 frames)
- Detailed error reporting with stack traces
- Summary statistics after processing

**Log Location:** `/volume1/docker/bird-id/logs/motion_detector.log`

### 3. Updated motion_detector.py
**New Features:**
- Loads settings from config.yaml
- CLI arguments override config (flexibility)
- Better error handling and logging
- Progress indicators during processing
- Graceful failure with informative messages

**Usage Priority:**
```
CLI arguments > Environment variables > config.yaml > hardcoded defaults
```

### 4. Dependencies Added
- **PyYAML>=6.0** - YAML configuration parsing

## Files Modified

- `config.yaml` - Created (central configuration)
- `src/motion_detector.py` - Enhanced with logging + config loading
- `requirements.txt` - Added PyYAML
- `README.md` - Updated with new workflow
- `setup.command` - One-time dependency installer
- `test-motion.command` - Easy motion testing script

## How to Use (NAS Execution)

### First Time Setup

```bash
# SSH into NAS
ssh -p 2000 vives@192.168.5.92

# Navigate to project
cd /volume1/docker/bird-id

# Run setup (installs dependencies)
bash setup.command
```

### Testing Motion Detection

**Method 1: Process a single clip**
```bash
ssh -p 2000 vives@192.168.5.92
cd /volume1/docker/bird-id
source venv/bin/activate

python src/motion_detector.py \
  --source "data/clips/cardinal/Birds 11-12-2025, 7.11.27am EST - 11-12-2025, 7.11.31am EST.mp4" \
  --output data/debug/test_run \
  --mask \
  --frames 0
```

**Method 2: Use config defaults**
```bash
# config.yaml has all settings
python src/motion_detector.py --mask --frames 200
```

**Method 3: Override specific settings**
```bash
python src/motion_detector.py \
  --diff-threshold 20 \
  --min-area 800 \
  --log-level DEBUG
```

### Checking Results

**View logs:**
```bash
tail -f /volume1/docker/bird-id/logs/motion_detector.log
```

**Check output frames:**
```bash
ls -lh /volume1/docker/bird-id/data/debug/test_run/
```

### Tuning Detection Sensitivity

Edit `/volume1/docker/bird-id/config.yaml`:

```yaml
motion:
  diff_threshold: 15    # Lower = more sensitive (10-30 typical)
  min_area: 500         # Smaller = detect smaller motions
  history: 30           # More frames = smoother background model
```

**Test different values:**
```bash
# Test with high sensitivity
python src/motion_detector.py --diff-threshold 10 --min-area 300

# Test with low sensitivity
python src/motion_detector.py --diff-threshold 25 --min-area 1000
```

## Next Steps

### 1. Dockerized Dual-Stream Service
- Wrap `dual_stream_detector.py` plus the new live-status UI into a container (patterned after `birdcam-live`).
- Integrate the daily RTSP refresh so Bird-ID always matches Frigate/BirdNET tokens.
- Add health checks, log rotation, and restart policies.

### 2. Live Dashboard
- Surface a live view + recent detections (species, confidence, snapshots) to monitor the classifier in real time.
- Ties into the roadmap Phase 2 deliverables.

### 3. EdgeTPU Compilation
- Compile the MobileNetV2 model for Coral to cut inference latency to <20 ms and free CPU.
- Update `classifier.py` to auto-select CPU vs TPU at runtime.

## Testing Checklist

- [ ] Run setup.command to install PyYAML
- [ ] Test with baseline clip (subtle motion)
- [ ] Test with cardinal clip (bird activity)
- [ ] Verify logs are being written
- [ ] Adjust thresholds and re-test
- [ ] Check output quality in debug frames

## Configuration Reference

### Current Settings (config.yaml)
```
diff_threshold: 15
min_area: 500
history: 30
roi: [(380,425), (1550,425), (1550,960), (380,960)]
```

### Recommended Starting Values
- **Outdoor daylight**: diff_threshold=15, min_area=500
- **Variable lighting**: diff_threshold=20, min_area=800
- **High sensitivity**: diff_threshold=10, min_area=300

## Troubleshooting

**Import Error: No module named 'yaml'**
```bash
source venv/bin/activate
pip install PyYAML
```

**No motion detected in clip with obvious motion**
- Lower diff_threshold (try 10)
- Lower min_area (try 300)
- Check if motion is inside ROI mask

**Too many false positives**
- Raise diff_threshold (try 25)
- Raise min_area (try 1000)
- Consider adding min_consecutive_frames logic

**Logs not appearing**
- Check logs directory exists: `mkdir -p logs`
- Verify logging.file is true in config.yaml
