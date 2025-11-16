# Video Testing Feature - Implementation Status

## Overview
Adding capability to test classifier with video files instead of live RTSP feed, with real-time UI controls for source selection and EdgeTPU toggle.

---

## ✅ COMPLETED

### 1. Video Conversion ✓
- **Script**: `scripts/convert_to_360p.sh`
- **Status**: Successfully converted all 33 videos to 640x360
- **Output**: `/data/clips_360p/` with organized subdirectories:
  - `baseline/` (4 videos)
  - `brownbird/` (4 videos)
  - `cardinal/` (2 videos)
  - `chickadees/` (10 videos)
  - `downy/` (2 videos)
  - `tuftedtitmouse/` (6 videos)
  - `whatisthis?/` (5 videos)

### 2. API Endpoints ✓
- **File**: `src/status_api.py`
- **Endpoints Added**:
  ```
  GET  /api/videos          - List all available test videos
  GET  /api/runtime-config  - Get current configuration
  POST /api/runtime-config  - Update configuration (source, video, TPU)
  ```
- **Features**:
  - Validates video files exist before saving config
  - Returns detailed video info (path, category, size)
  - Persists settings to `runtime_config.json`

### 3. Runtime Configuration System ✓
- **File**: `data/runtime_config.json`
- **Schema**:
  ```json
  {
    "source_mode": "live|video",
    "video_file": "relative/path/to/video.mp4",
    "use_edgetpu": true|false,
    "last_updated": "ISO timestamp"
  }
  ```

### 4. Source Manager Module ✓
- **File**: `src/source_manager.py`
- **Functions**:
  - `load_runtime_config()` - Load runtime settings
  - `get_source_urls(config)` - Return appropriate sources (RTSP or video)
  - `should_use_edgetpu()` - Check EdgeTPU setting
- **Logic**:
  - For RTSP: Returns separate sub/main streams
  - For video: Returns same video for both (single source)
  - Falls back to defaults if config missing

### 5. Configuration Updates ✓
- **File**: `config.yaml`
- **Added**:
  - `video:` section with clips_dir and default video
  - Documentation for source selection

### 6. Docker Fixes ✓
- **File**: `requirements.txt`
- Fixed `pycoral` version constraint (was >=2.0, now just `pycoral`)

---

### 7. Source Manager Integration ✓
- **Files**: `src/dual_stream_detector_v2.py`, `src/dual_stream_detector_live.py`
- **Changes Made**:
  - Added source_manager import with fallback handling
  - Replaced direct RTSP config reading with `get_source_urls(config)`
  - Both detectors now support RTSP and video file sources seamlessly

### 8. Web UI Controls ✓
- **File**: `share/index.html`
- **Added**:
  - Source mode selector (Live/Video dropdown)
  - Video file picker (dynamically populated from API)
  - EdgeTPU toggle checkbox
  - Apply & Restart button with status messages
  - Responsive styling matching existing design
- **JavaScript Functions**:
  - `loadCurrentConfig()` - Loads runtime config on page load
  - `loadVideoList()` - Fetches and populates video options
  - Source mode change handler - Shows/hides video selector
  - Apply button handler - POSTs config to API

---

## ⏳ IN PROGRESS / TODO

### Priority 1: Build & Test
1. **Rebuild Docker image** (after integrating source manager):
   ```bash
   cd /volume1/docker
   sudo /usr/local/bin/docker-compose build birdid
   ```

2. **Test video mode**:
   ```bash
   # Update runtime config via API or manually
   echo '{"source_mode":"video","video_file":"chickadees/Birds 11-12-2025, 3.54.32pm EST - 11-12-2025, 3.54.37pm EST.mp4","use_edgetpu":true}' > /volume1/docker/bird-id/data/runtime_config.json

   # Restart container
   sudo /usr/local/bin/docker-compose restart birdid

   # Check logs
   sudo /usr/local/bin/docker-compose logs -f birdid | grep -i "video\|source"
   ```

3. **Test EdgeTPU toggle**:
   ```bash
   # Toggle TPU off
   # Update config with use_edgetpu: false
   # Restart and verify CPU mode in logs
   ```

4. **Test UI**:
   - Open https://birdid.vivessyn.duckdns.org
   - Switch to video mode
   - Select a video
   - Toggle EdgeTPU
   - Click Apply
   - Manually restart container
   - Verify video playback and classification

---

## Future Enhancements

1. **Auto-restart** - Add Docker API endpoint to restart container from UI
2. **Video looping options** - Configure loop count or one-shot playback
3. **Playlist mode** - Queue multiple videos for sequential testing
4. **Frame rate control** - Adjust playback speed for debugging
5. **Side-by-side comparison** - View CPU vs TPU classification simultaneously

---

## Quick Reference

### Test a specific video manually:
```bash
# Via runtime config
echo '{"source_mode":"video","video_file":"downy/Birds 11-10-2025, 8.09.23am EST - 11-10-2025, 8.09.28am EST.mp4","use_edgetpu":true}' \
  > /volume1/docker/bird-id/data/runtime_config.json

docker-compose restart birdid
```

### Switch back to live mode:
```bash
echo '{"source_mode":"live","video_file":null,"use_edgetpu":true}' \
  > /volume1/docker/bird-id/data/runtime_config.json

docker-compose restart birdid
```

### List available videos via API:
```bash
curl http://localhost:8000/api/videos | jq '.videos[] | "\(.category)/\(.name)"'
```

---

**Last Updated**: 2025-11-15
**Status**: Core infrastructure complete, integration in progress
