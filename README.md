# Bird-ID Workspace

Motion detection and bird species classification for the backyard feeder cam.

## 🚀 Quick Start (Double-Click Workflow)

### First Time Setup

1. **Double-click** `setup.command`
2. Wait for "Setup Complete!"
3. Done!

### Test Motion Detection

**Drag & Drop (easiest):**
- Drag an MP4 file onto `test-motion.command`
- Results appear in `data/clips/output/`

**Or use the inbox folder:**
- Copy MP4 files to `data/clips/inbox/`
- Double-click `test-motion.command`
- Check `data/clips/output/` for results

### Adjust Settings

Edit `config.yaml` to tune detection:
- `diff_threshold: 15` - Lower = more sensitive
- `min_area: 500` - Minimum motion size (pixels)
- `roi:` - Region of interest coordinates

## 📁 Project Layout

```
bird-id/
├── setup.command           ← One-time setup (double-click once)
├── test-motion.command     ← Motion testing (drag videos here)
├── config.yaml             ← Settings (edit to tune)
├── src/                    ← Python code
│   ├── motion_detector.py
│   ├── stream_reader.py
│   └── classifier.py (coming soon)
├── data/
│   ├── clips/
│   │   ├── inbox/          ← Drop test videos here
│   │   └── output/         ← Results appear here
│   └── events/             ← Live detection events
├── logs/                   ← Check motion_detector.log for details
└── docs/                   ← Planning & notes
```

## 🔧 For Developers

**Command-line usage:**

```bash
# Activate environment
source venv/bin/activate

# Test motion detection
python src/motion_detector.py --source clip.mp4 --mask --log-level DEBUG

# Grab RTSP snapshot
python src/stream_reader.py --snapshot

# Override config settings
python src/motion_detector.py --diff-threshold 20 --min-area 800
```

## 🐦 Live Dual-Stream Detection & Classification

The production pipeline runs the substream for fast motion detection and the mainstream for MobileNetV2 bird classification.

```bash
ssh -p 2000 vives@192.168.5.92
cd /volume1/docker/bird-id
source venv/bin/activate

python src/dual_stream_detector.py \
  --classify \
  --frames 0 \
  --frame-skip 5
```

This script:
- Watches both RTSP streams simultaneously
- Stores crops + metadata in `data/birds.db`
- Logs detections to `logs/dual_stream_detector.log`

See `CLASSIFICATION_SYSTEM.md` for the full architecture and database queries.

## 🐳 Docker + Live View

The dual-stream detector now ships as a container (patterned after `birds-live`) so you can watch the classifier output from Traefik.

```bash
cd /volume1/docker
docker compose up -d birdid birdid-share
```

Then open `https://birdid.vivessyn.duckdns.org` for the annotated HLS feed rendered directly by the classifier. Segments live under `bird-id/share/hls`, logs under `bird-id/logs`, and detections/SQLite data remain in `bird-id/data`.

The same domain also exposes JSON APIs for the future UI:

- `https://birdid.vivessyn.duckdns.org/api/status`
- `https://birdid.vivessyn.duckdns.org/api/recent`

These endpoints are served by the new FastAPI status service that runs alongside the detector inside the `birdid` container.

### EdgeTPU Acceleration

To offload classification to the Coral USB accelerator, compile the TFLite model once (from the NAS or any Docker host) and point the container at the EdgeTPU version:

```bash
docker run --rm \
  -v /volume1/docker/bird-id/data/models:/models \
  gcr.io/coral-project/edgetpu-compiler \
  edgetpu_compiler /models/birds_v1.tflite -o /models/

# or ./scripts/compile_edgetpu.sh /volume1/docker/bird-id/data/models
```

Then set `BIRDID_USE_EDGETPU=1` (and optionally `BIRDID_MODEL_PATH=/data/models/birds_v1_edgetpu.tflite`) in `docker-compose.yaml` and restart `birdid`.

See `docs/PROJECT_OVERVIEW.md` + `docs/ROADMAP.md` for how this service plugs into the rest of the system and what’s coming next (EdgeTPU, live dashboard widgets, etc.).

## 📚 Documentation Hub

- `docs/PROJECT_OVERVIEW.md` – Current system diagram, key scripts, how data flows.
- `CLASSIFICATION_SYSTEM.md` – Deep dive into the dual-stream + classifier stack.
- `docs/ROADMAP.md` – Phase-by-phase delivery plan (Docker service, UI, database, etc.).
- `docs/FUTURE_VISION.md` – Experience ideas (seed mix tracking, highlight reels, feather forecast).

Use these docs to understand the state of the project today and where it is headed next.
