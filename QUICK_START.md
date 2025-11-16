# Quick Start Guide - NAS Execution

## One-Time Setup

```bash
ssh -p 2000 vives@192.168.5.92
cd /volume1/docker/bird-id
bash setup.command
```

Wait for "Setup Complete!"

## Run a Test

```bash
ssh -p 2000 vives@192.168.5.92
cd /volume1/docker/bird-id
source venv/bin/activate

# Test on existing cardinal clip
python src/motion_detector.py \
  --source "data/clips/cardinal/Birds 11-12-2025, 7.11.27am EST - 11-12-2025, 7.11.31am EST.mp4" \
  --output data/debug/cardinal_test \
  --mask \
  --frames 0
```

## Check Results

```bash
# View the log
tail -50 logs/motion_detector.log

# Count output frames
ls data/debug/cardinal_test/*.jpg | wc -l

# View a sample frame (copy to Mac to view)
# Frame files are at: data/debug/cardinal_test/frame_XXXX.jpg
```

## Tune Settings

Edit `config.yaml`:
```bash
nano config.yaml
```

Change these values:
- `diff_threshold: 15` → Lower for more sensitivity
- `min_area: 500` → Lower to detect smaller motion

Save and re-run the test.

## Common Commands

```bash
# SSH shortcut (save this)
alias birdnas='ssh -p 2000 vives@192.168.5.92'

# Quick test run
birdnas "cd /volume1/docker/bird-id && source venv/bin/activate && python src/motion_detector.py --mask --frames 100"

# View recent log
birdnas "tail -50 /volume1/docker/bird-id/logs/motion_detector.log"
```
