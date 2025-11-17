# Clip-Based Motion Test

Commands used:
```bash
source /volume1/docker/bird-id/venv/bin/activate
python /volume1/docker/bird-id/src/motion_detector.py --source "/volume1/docker/bird-id/data/clips/baseline/some movement.mp4" --output /volume1/docker/bird-id/data/debug/clips_baseline --frames 0
python /volume1/docker/bird-id/src/motion_detector.py --source "/volume1/docker/bird-id/data/clips/cardinal/Birds 11-12-2025, 7.11.27am EST - 11-12-2025, 7.11.31am EST.mp4" --output /volume1/docker/bird-id/data/debug/clips_cardinal --frames 0
```

Outputs:
- Baseline debug frames in `/volume1/docker/bird-id/data/debug/clips_baseline/`
- Cardinal activity frames in `/volume1/docker/bird-id/data/debug/clips_cardinal/`

Use these folders to tune thresholds before running live stream tests.
