# Google Coral Edge TPU Setup Guide

## Overview
Successfully configured Google Coral USB Accelerator for real-time bird detection, achieving **17 FPS average** (10-27 FPS range) vs 0.6 FPS with YOLO on CPU - a **28x performance improvement**.

## Hardware
- **Device**: Google Coral USB Accelerator (Edge TPU)
- **Vendor ID**: 18d1 (Google)
- **Product ID**: 9302
- **Host**: Synology NAS (DSM 7.x)
- **Container**: Docker with docker-compose

## Package Versions (CRITICAL - Must Match)

```dockerfile
# Debian/Ubuntu packages
libedgetpu1-std: 16.0

# Python packages (from Google's repository)
pycoral: 2.0.0
tflite-runtime: 2.5.0.post1
```

**Installation from Google's repository:**
```dockerfile
# Add Coral Edge TPU repository
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && \
    apt-get install -y libedgetpu1-std

# Install pycoral and tflite-runtime with matching versions
RUN pip install --no-cache-dir \
    --extra-index-url https://google-coral.github.io/py-repo/ \
    pycoral~=2.0
```

## Docker Configuration (REQUIRED)

### docker-compose.yaml
```yaml
services:
  birdid-yolo:
    container_name: birdid-yolo
    privileged: true  # CRITICAL - Required for Edge TPU access
    devices:
      - /dev/bus/usb:/dev/bus/usb  # USB device mapping
    # ... rest of config
```

**Why `privileged: true` is required:**
- Edge TPU needs low-level hardware access
- USB device delegate loading requires elevated permissions
- Frigate and other Coral implementations use this same approach

**Security Note:** For production, can replace with specific permissions:
```yaml
security_opt:
  - apparmor=unconfined
  - systempaths=unconfined
devices:
  - /dev/bus/usb
volumes:
  - /sys/:/sys/:ro
```

## Host System Requirements

### 1. Udev Rules (Required on Host)
Create `/lib/udev/rules.d/99-edgetpu.rules`:
```bash
# Google Coral Edge TPU
SUBSYSTEM=="usb", ATTRS{idVendor}=="1a6e", GROUP="users", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", GROUP="users", MODE="0666"
```

Apply rules:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb
```

Verify device permissions:
```bash
ls -la /dev/bus/usb/002/003
# Should show: crw-rw-rw- 1 root users
```

### 2. Verify Edge TPU Detection
```bash
# Check USB device
lsusb | grep "Google"
# Output: Bus 002 Device 003: ID 18d1:9302 Google Inc.

# Check sysfs
cat /sys/bus/usb/devices/2-3/idVendor  # Should show: 18d1
cat /sys/bus/usb/devices/2-3/idProduct # Should show: 9302
```

## Common Issues & Solutions

### Issue 1: "Failed to load delegate from libedgetpu.so.1"

**Causes & Solutions:**

1. **Concurrent Access** (Most Common)
   - **Problem**: Another container/process is using the TPU
   - **Solution**: Stop other containers accessing the Edge TPU
   ```bash
   docker stop birdid  # Stop conflicting container
   docker restart birdid-yolo
   ```

2. **Missing Privileged Mode**
   - **Problem**: Container lacks hardware access permissions
   - **Solution**: Add `privileged: true` to docker-compose.yaml
   ```yaml
   services:
     your-service:
       privileged: true
   ```

3. **USB Connection Issue**
   - **Problem**: Unstable USB connection
   - **Solution**: Physically replug the USB device
   ```bash
   # After replug, restart container
   docker-compose restart birdid-yolo
   ```

4. **Version Mismatch**
   - **Problem**: Incompatible package versions
   - **Solution**: Install from Google's repository (see above)
   - **DO NOT** install pycoral from PyPI - use Google's repo only

5. **Missing Udev Rules**
   - **Problem**: Device permissions not set
   - **Solution**: Create udev rules (see Host System Requirements)

### Issue 2: Python Import Errors

**Error**: `ModuleNotFoundError: No module named 'pycoral.adapters'`

**Cause**: Wrong pycoral package from PyPI

**Solution**:
```dockerfile
# WRONG - PyPI version
RUN pip install pycoral

# CORRECT - Google's repository
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

## Python Code Example

```python
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
import tflite_runtime.interpreter as tflite

# Initialize Edge TPU
try:
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    print("Edge TPU initialized successfully")
except ValueError as e:
    if "Failed to load delegate" in str(e):
        print("Edge TPU not available, check:")
        print("1. Is another process using the TPU?")
        print("2. Is container running with privileged: true?")
        print("3. Is USB device properly connected?")
```

## Performance Results

| Metric | YOLO (CPU) | TFLite (Edge TPU) | Improvement |
|--------|-----------|-------------------|-------------|
| FPS    | 0.6       | 17 (avg)          | 28x faster  |
| Latency| ~1667ms   | ~59ms             | 28x faster  |
| Hardware| CPU only  | Edge TPU          | Dedicated ML accelerator |

## Verification Checklist

- [ ] libedgetpu1-std installed from Google's apt repository
- [ ] pycoral 2.0.0 installed from Google's pip repository
- [ ] tflite-runtime 2.5.0.post1 installed (dependency of pycoral)
- [ ] Udev rules created on host system
- [ ] `privileged: true` in docker-compose.yaml
- [ ] `/dev/bus/usb` mapped to container
- [ ] No other containers using the Edge TPU
- [ ] USB device physically connected and detected
- [ ] Python imports succeed without errors
- [ ] Interpreter creates successfully with "Using Edge TPU" message

## Debugging Commands

```bash
# Check if TPU is accessible in container
docker exec birdid-yolo ls -la /dev/bus/usb/002/003

# Check installed packages
docker exec birdid-yolo pip list | grep -E "(pycoral|tflite)"
docker exec birdid-yolo dpkg -l | grep edgetpu

# Test Edge TPU initialization
docker exec birdid-yolo python3 -c "
from pycoral.utils.edgetpu import make_interpreter
interpreter = make_interpreter('/data/models/ssd_mobilenet_v2_edgetpu.tflite')
print('SUCCESS: Edge TPU working')
"

# Check for concurrent access
docker ps | grep -E "birdid|frigate"  # Look for conflicting containers

# Monitor detection performance
tail -f /volume1/docker/bird-id-yolo/logs-yolo/tflite_detection.log | grep FPS
```

## References

- Google Coral Documentation: https://coral.ai/docs/
- PyCoral Python API: https://coral.ai/docs/reference/py/
- Edge TPU Models: https://coral.ai/models/
- Frigate NVR (reference implementation): https://github.com/blakeblackshear/frigate

## Troubleshooting Resources

- GitHub Issue: "Failed to load delegate" is usually hardware access, not software
- Stack Overflow: USB replug often resolves connection issues
- Only ONE process can access the Edge TPU at a time
- Always use Google's official repositories, not PyPI for pycoral
