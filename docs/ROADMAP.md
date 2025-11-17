# Bird-ID Delivery Roadmap

_Last updated: 2025-11-14_

## Phase 1 – Foundations ✅
- Centralized config (`config.yaml`)
- Reliable MOG2 motion detection on the UniFi substream
- Dual-stream classifier prototype (CPU mode)
- SQLite event storage + logging

Status: Complete (running manually on the NAS)

## Phase 2 – Service Hardening (ACTIVE)
1. **Dockerized Dual-Stream Service** ✅  
   - `birdid` container (built from `./bird-id`) runs the detector + annotated preview pipeline via `scripts/run_service.sh`.  
   - RTSP auto-refresh stays in place (`refresh_unifi_streams.sh`).  
   - `birdid-share` (nginx) serves `/hls/birdid.m3u8` through Traefik (`birdid.vivessyn.duckdns.org`).
2. **EdgeTPU Compilation**  
   - Compile `birds_v1.tflite` for Coral → <20 ms inference.  
   - Update `classifier.py` to auto-detect TPU.
3. **Live Status UI** (IN PROGRESS)  
   - ✅ FastAPI status API (`/api/status`, `/api/recent`) runs inside the `birdid` container.  
   - ⏳ Next: surface the data in a dashboard (graphs, tables, controls) and integrate with the HLS viewer.

## Phase 3 – Calibration & Personalization
- First-day onboarding (Yes/No confirmations, ROI guidance, “no bird” baseline).
- Automatic scale normalization (learn feeder geometry, lighting patterns).
- User correction loop feeding micro-training / weighting.

## Phase 4 – Delightful Visualizations
- Density graphs with species-themed bars and icons.
- Seed mix logging + popularity metrics.
- Seasonal/weather UI themes.
- Snapshot gallery with filters.

## Phase 5 – Digest & Highlight Reels
- Mario-Kart style recap video generator (per species / per time range).
- Export/share workflows.

## Phase 6 – Predictive & Intelligent Features
- “Feather Forecast” – combine history + weather.  
  Example: “Tomorrow 78 % chickadee likelihood, peak 7:10 AM.”
- Feeder optimization suggestions (seed mix impact).

## Phase 7 – Enrichment & Community (Optional)
- Bird lore mode / bird journal entries.
- Community sharing or weekly “top sightings.”
- Privacy-aware publishing of highlight reels.

## Work Tracking
- Implementation log: `docs/IMPLEMENTATION_LOG.md`
- Classification deep dive: `CLASSIFICATION_SYSTEM.md`
- Vision & inspiration: `docs/FUTURE_VISION.md`

Use this roadmap to decide what lands in the Docker service today versus which features move into future iterations.
