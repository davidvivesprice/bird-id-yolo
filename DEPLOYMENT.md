# Bird-ID Deployment Guide

## Automatic EdgeTPU Detection

Bird-ID **automatically detects and uses** your Coral EdgeTPU if available, falling back to CPU mode if not. No configuration needed!

### Performance Comparison

| Mode | Inference Time | CPU Usage | Hardware |
|------|---------------|-----------|----------|
| **EdgeTPU** | ~10ms | 10-20% | Coral USB ($60) |
| **CPU Fallback** | ~100ms | 40-60% | Any system |

---

## Quick Start

### 1. Build and Deploy

```bash
cd /volume1/docker

# Build the image
sudo /usr/local/bin/docker-compose build birdid

# Start services
sudo /usr/local/bin/docker-compose up -d birdid birdid-share

# Check logs
sudo /usr/local/bin/docker-compose logs -f birdid
```

### 2. Verify It's Working

**Check container status:**
```bash
sudo /usr/local/bin/docker ps | grep birdid
```

**Check if EdgeTPU is being used:**
```bash
sudo /usr/local/bin/docker-compose logs birdid | grep -i "edgetpu\|tpu\|coral"
```

You should see:
- ✅ **With Coral TPU**: "EdgeTPU delegate loaded successfully"
- ⚙️ **Without Coral TPU**: "EdgeTPU not available; falling back to CPU"

**Access the web interface:**
```
https://birdid.vivessyn.duckdns.org
```

The status page shows whether EdgeTPU is active or CPU fallback is being used.

---

## Model Files

The EdgeTPU model automatically falls back to CPU if no TPU is detected:

```bash
ls -lh /volume1/docker/bird-id/data/models/birds_v1_edgetpu.tflite
# Should be ~4.1 MB, dated 2025-11-15 or later

/volume1/docker/bird-id/data/models/inat_bird_labels.txt
# Label file with bird species names
```

---

## Toggling EdgeTPU On/Off

You can disable EdgeTPU and force CPU-only mode by changing the environment variable:

**Edit `/volume1/docker/docker-compose.yaml`:**
```yaml
environment:
  - BIRDID_USE_EDGETPU=0  # Change from 1 to 0 to force CPU mode
```

**Restart the service:**
```bash
sudo /usr/local/bin/docker-compose restart birdid
```

This is useful for:
- Testing performance differences
- Debugging classification issues
- Comparing CPU vs TPU inference

---

## Troubleshooting

### EdgeTPU Not Detected

**Symptoms:**
```
Warning: EdgeTPU delegate not available; falling back to CPU
```

**Solutions:**
1. Verify Coral TPU is plugged into USB port
2. Check USB device is accessible:
   ```bash
   ls -l /dev/bus/usb/*/*
   ```
3. Ensure docker-compose.yaml has `devices:` section
4. Try different USB port
5. Check Coral LED is lit (should be solid or blinking)

### CPU Mode Too Slow

**Symptoms:**
- High CPU usage (>80%)
- Frame processing lag
- Delayed detections

**Solutions:**
1. Increase `BIRDID_FRAME_SKIP` from 5 to 10 (process fewer frames)
2. Reduce `BIRDID_OUTPUT_FPS` from 10 to 5 (lower output framerate)
3. Lower video resolution in config.yaml
4. **Best solution:** Get a Coral TPU!

### Wrong Model File

**Symptom (EdgeTPU mode):**
```
RuntimeError: Encountered unresolved custom op: edgetpu-custom-op
```
This actually means the model IS an EdgeTPU model, but the TPU device isn't available. See "EdgeTPU Not Detected" above.

**Symptom (CPU mode with EdgeTPU model):**
```
RuntimeError: Encountered unresolved custom op: edgetpu-custom-op
```
Solution: Change `BIRDID_MODEL_PATH` to `/data/models/birds_v1.tflite` (remove `_edgetpu`)

---

## Configuration Details

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIRDID_USE_EDGETPU` | `1` | Enable EdgeTPU (set to `0` to force CPU mode) |
| `BIRDID_MODEL_PATH` | `birds_v1_edgetpu.tflite` | Model file (works with or without TPU) |
| `BIRDID_FRAME_SKIP` | `5` | Process every Nth frame (increase to reduce CPU) |
| `BIRDID_OUTPUT_FPS` | `10` | HLS output framerate |
| `BIRDID_STATUS_API` | `1` | Enable REST API for status/statistics |
| `BIRDID_STATUS_PORT` | `8000` | Port for status API |

### How Auto-Detection Works

The classifier automatically detects EdgeTPU availability:

1. If `BIRDID_USE_EDGETPU=1` and Coral TPU is connected → **Use EdgeTPU**
2. If `BIRDID_USE_EDGETPU=1` but no Coral TPU found → **Fall back to CPU**
3. If `BIRDID_USE_EDGETPU=0` → **Force CPU mode** (even if TPU available)

---

## Recommended Settings

**For production (24/7 monitoring with Coral TPU):**
```yaml
- BIRDID_USE_EDGETPU=1
- BIRDID_FRAME_SKIP=5
- BIRDID_OUTPUT_FPS=10
```

**For testing/development (CPU only):**
```yaml
- BIRDID_USE_EDGETPU=0          # Force CPU mode for testing
- BIRDID_FRAME_SKIP=10           # Process fewer frames
- BIRDID_OUTPUT_FPS=5            # Lower bandwidth
```

**For low-power systems (Raspberry Pi with Coral TPU):**
```yaml
- BIRDID_USE_EDGETPU=1           # Offload to TPU
- BIRDID_FRAME_SKIP=10           # Reduce processing load
- BIRDID_OUTPUT_FPS=5            # Lower bandwidth
```

---

## Build Time Expectations

First build (same for both modes):
- **5-15 minutes** depending on internet speed
- Downloads Python 3.9 base image (~150 MB)
- Installs system packages (ffmpeg, opencv, etc.)
- Installs Python packages (tensorflow-lite, pycoral, etc.)

Subsequent builds:
- **< 1 minute** due to Docker layer caching

---

## Health Checks

**Check if EdgeTPU is being used:**
```bash
# Look for EdgeTPU initialization in logs
sudo /usr/local/bin/docker-compose logs birdid | grep -i edgetpu

# Expected output (TPU mode):
"EdgeTPU delegate loaded successfully"

# Expected output (CPU mode):
"Using CPU for inference" OR no EdgeTPU messages
```

**Monitor CPU usage:**
```bash
# TPU mode should use 10-20% CPU
# CPU mode will use 40-60% CPU
sudo /usr/local/bin/docker stats birdid
```

**Check inference speed:**
```bash
# Watch for classification time in debug logs
sudo /usr/local/bin/docker-compose logs -f birdid | grep "classification"
```

---

## Getting Coral TPU

If you want to upgrade to EdgeTPU mode:

**Purchase:**
- [Google Coral USB Accelerator](https://coral.ai/products/accelerator) - $59.99
- Amazon, Adafruit, or other retailers

**Compatibility:**
- Works on any system with USB port
- Supported on Linux, Windows, macOS
- No additional drivers needed in Docker

**Installation:**
1. Plug Coral into USB port
2. Verify it's detected: `ls /dev/bus/usb/*/*`
3. Use `birdid-tpu` mode in docker-compose.yaml
4. Restart containers

---

## Future Roadmap

### Ultimate Goal: Turnkey SBC Appliance

The long-term vision for Bird-ID is to become a **zero-CLI, plug-and-play appliance** for Single Board Computers (Raspberry Pi, Orange Pi, etc.). This is currently in the planning phase.

**Target User Experience:**
1. Flash pre-configured SD card image to SBC
2. Plug in power, camera, and (optionally) Coral EdgeTPU
3. Open web browser to `http://birdid.local`
4. Configure camera RTSP URL via web UI
5. Done - bird identification starts automatically

**Technical Approach (Planned):**
- Use Docker internally (invisible to end users)
- Systemd auto-start on boot
- Web-based first-run configuration wizard
- Auto-detection of Coral EdgeTPU
- Avahi/mDNS for `birdid.local` hostname
- OTA (Over-The-Air) updates via web UI
- Pre-built SD card images for common SBC platforms

**Why This Matters:**
- Makes bird identification accessible to non-technical users
- Removes all command-line requirements
- Creates a true "appliance" experience
- Enables wider adoption for citizen science, education, conservation

**Current Status:** Planning phase - focus is on polishing the Docker deployment first

See `install.sh` for early proof-of-concept automated installer.

---

*For more information, see [SYSTEM_DOCUMENTATION.md](SYSTEM_DOCUMENTATION.md)*
