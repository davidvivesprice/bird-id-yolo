# Bird-ID Project Overview

_Last updated: 2025-11-14_

## 1. Purpose

Bird-ID processes the backyard UniFi feeder cam to:

1. Run **substream (640×360)** video through fast MOG2 motion detection.
2. Crop aligned regions from the **mainstream (1920×1080)** whenever motion is detected.
3. Classify the crop with a MobileNetV2 TFLite model (CPU today, EdgeTPU soon).
4. Store detections, crops, and metadata in SQLite for dashboards + experiments.

The goal is a reliable, self-contained service that can also expose a live “birds-live” style view inside Docker.

## 2. Key Scripts

| Path | Purpose |
| --- | --- |
| `src/dual_stream_detector.py` | **Primary runtime** – opens both RTSP streams, runs detection on the substream, classification on the mainstream, writes events to SQLite/logs. |
| `src/motion_detector_mog2.py` | Standalone motion detector (used by the dual-stream runner). |
| `src/classifier.py` | TFLite wrapper for the 965-species MobileNetV2 model (CPU + EdgeTPU ready). |
| `src/bird_database.py` | SQLite helper for sightings, queries, and stats. |
| `src/stream_reader.py` | Snapshot utility for testing RTSP connectivity and ROI tuning. |
| `setup.command` / `test-motion.command` | Double-click entry points for macOS/NAS users. |

Supporting scripts (`extract_test_samples.py`, `test_accuracy.py`, etc.) live under `src/` for regression testing.

## 3. Configuration & Data

- `config.yaml` – Centralized RTSP URLs (auto-refreshed daily), motion parameters, directories.
- `data/`
  - `clips/` – Baseline clips for regression tests.
  - `events/` – Snapshots + metadata written by the live detector.
  - `birds.db` – SQLite sightings database.
- `logs/`
  - `motion_detector.log`, `dual_stream_detector.log`, etc. (rotating).

## 4. Current Capabilities

- ✅ Dual-stream detection + classification running on the NAS CPU.
- ✅ Daily RTSP auto-refresh (Frigate/BirdNET/Bird-ID stay in sync).
- ✅ CLI + double-click workflows for debugging clips.
- ✅ Documentation for classifier internals (`CLASSIFICATION_SYSTEM.md`) and testing.

## 5. Near-Term Priorities

1. **Docker Service (birds-live style)** ✅  
   `birdid` now runs the dual-stream detector + annotated preview pipeline, writing HLS segments served by `birdid-share` (Traefik host `birdid.vivessyn.duckdns.org`). `scripts/run_service.sh` supervises both processes and logs.

2. **EdgeTPU Compilation**  
   Compile `birds_v1.tflite` for Coral to unlock <20 ms inference.

3. **Live Visualization**  
   Build on the Docker service: expose recent detections, classifier stats, and SQLite queries via a lightweight dashboard/API.

Roadmap details live in `docs/ROADMAP.md`.

## 6. How to Run

```bash
ssh -p 2000 vives@192.168.5.92
cd /volume1/docker/bird-id
source venv/bin/activate

python src/dual_stream_detector.py --classify --frames 0 --frame-skip 5
```

Use `CTRL+C` to stop. Logs are written to `logs/dual_stream_detector.log`; detections land in `data/birds.db`.

### Docker / Traefik service

```bash
cd /volume1/docker
docker compose up -d birdid birdid-share
```

Live preview: `https://birdid.vivessyn.duckdns.org` (Traefik).  
Status APIs: `/api/status`, `/api/recent` (proxied to FastAPI running inside the container).

## 7. Future Experience

For inspiration (seed mix analytics, highlight reels, feather forecasts, etc.), see `docs/FUTURE_VISION.md`. That document is intentionally aspirational—it feeds the main roadmap when we are ready to tackle the UX layers.
