# Bird-ID Workspace (Vivessyn)

## Paths
- Root: /volume1/docker/bird-id
- Source code: /volume1/docker/bird-id/src
- Data & events: /volume1/docker/bird-id/data
- Logs: /volume1/docker/bird-id/logs
- Docs: /volume1/docker/bird-id/docs
- Experimental share assets: /volume1/docker/bird-id/share

## Python Environment
- Virtualenv: /volume1/docker/bird-id/venv
- Requirements: /volume1/docker/bird-id/requirements.txt

Activate & install:
  source /volume1/docker/bird-id/venv/bin/activate
  pip install -U pip
  pip install -r /volume1/docker/bird-id/requirements.txt

## HLS Share Duplicate
A snapshot of the current public HLS assets lives at /volume1/docker/bird-id/share/birds-share-base.
This mirrors /volume1/docker/birdcam-live/birds-share (HTML, nginx config, structure) so we can copy it for experiments without touching the live stream.

## Next Steps
1. Implement RTSP frame capture under src/stream_reader.py.
2. Build motion detection + classifier modules per docs/bird_species_classifier_plan.md.
3. When testing, copy share/birds-share-base into a new folder and wire separate compose services so production birds-hls / birds-share stay untouched.
