# EdgeTPU Compilation - Critical Pipeline Issue

**Status:** ⚠️ UNRESOLVED - CRITICAL FOR PRODUCTION
**Date:** 2025-11-14
**Priority:** HIGH

---

## Problem Statement

The bird classification model (`birds_v1.tflite`) currently runs on CPU. For production 24/7 operation with real-time classification, **EdgeTPU acceleration is essential**.

**Performance Impact:**
- CPU inference: ~100-200ms per image
- EdgeTPU inference: ~10-20ms per image (10x faster)
- With birds visiting every few minutes, EdgeTPU enables instant classification without lag

---

## Current Status

### ✅ What's Working
- TFLite model downloaded: `data/models/birds_v1.tflite` (3.5 MB)
- Labels file: `data/models/inat_bird_labels.txt` (965 species)
- CPU inference tested and working
- pycoral library installed
- Coral USB Accelerator available (shared with Frigate)

### ❌ What's Blocking
- **EdgeTPU compiler not available on Synology NAS**
- Compiler requires Debian-based Linux with x86_64
- Synology DSM is not standard Debian (custom package manager)
- Direct compiler download failed (404 error on GitHub release)

---

## Technical Requirements

### EdgeTPU Compiler Specs
- **Tool:** `edgetpu_compiler`
- **Platform:** 64-bit Debian 6.0+ or Ubuntu 18.04+
- **Purpose:** Converts TFLite model → EdgeTPU-compatible `.tflite` file
- **Output:** `birds_v1_edgetpu.tflite` (compiled for Coral USB)

### Model Requirements
- Input TFLite model must be quantized (uint8)
- Our model: `birds_v1.tflite` (already quantized ✅)
- Input shape: [1, 224, 224, 3]
- Output shape: [1, 965]

---

## Solution Options

### Option 1: Docker Container Compilation (RECOMMENDED)
Run EdgeTPU compiler in Docker container on NAS:

```bash
# Pull Google Coral compiler container
docker run --rm \
  -v /volume1/docker/bird-id/data/models:/models \
  google/edgetpu-compiler:latest \
  edgetpu_compiler /models/birds_v1.tflite -o /models/
```

**Pros:**
- Doesn't require Synology package installation
- Official Google container
- Portable solution

**Cons:**
- Requires Docker (already available ✅)
- One-time setup

### Option 2: Compile on Mac, Transfer to NAS
Install compiler on Mac (if compatible) and transfer compiled model:

```bash
# On Mac (if x86_64 or through Rosetta)
edgetpu_compiler birds_v1.tflite
# Transfer to NAS
scp birds_v1_edgetpu.tflite vives@192.168.5.92:/volume1/docker/bird-id/data/models/
```

**Pros:**
- Clean separation of concerns
- Mac has better tooling support

**Cons:**
- Mac is ARM64 (M-series) - compiler may not work
- Extra manual step in pipeline

### Option 3: GitHub Actions CI/CD Pipeline
Create automated compilation pipeline:

```yaml
# .github/workflows/compile-model.yml
on:
  workflow_dispatch:

jobs:
  compile:
    runs-on: ubuntu-latest
    steps:
      - name: Install EdgeTPU Compiler
        run: |
          curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
          echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
          sudo apt-get update
          sudo apt-get install edgetpu-compiler

      - name: Compile Model
        run: edgetpu_compiler birds_v1.tflite

      - name: Upload Artifact
        uses: actions/upload-artifact@v3
        with:
          name: edgetpu-model
          path: birds_v1_edgetpu.tflite
```

**Pros:**
- Automated, reproducible
- No local dependencies
- Version controlled

**Cons:**
- Requires GitHub repo setup
- Extra complexity for one-time task

### Option 4: Use Pre-compiled EdgeTPU Model
Search for pre-compiled bird classification models:

```bash
# Check Google Coral model zoo
wget https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite
```

**Pros:**
- Instant solution if available
- No compilation needed

**Cons:**
- May not be same model version
- May not have same species labels

---

## Recommended Action Plan

### Immediate (Today):
1. **Try Option 1 (Docker)** - Most pragmatic solution
   ```bash
   ssh -p 2000 vives@192.168.5.92
   docker run --rm -v /volume1/docker/bird-id/data/models:/models \
     gcr.io/coral-project/edgetpu-compiler \
     edgetpu_compiler /models/birds_v1.tflite -o /models/
   ```

2. **If Docker fails, try Option 4** - Check for pre-compiled models
   - Search Google Coral model zoo
   - Verify species list matches our labels

3. **Fallback to CPU for now** - Continue development
   - Classification works on CPU
   - Add EdgeTPU support as conditional
   - Production deployment can wait for compilation

### Short-term (This Week):
1. Resolve compilation using Docker or GitHub Actions
2. Test EdgeTPU model on Coral USB
3. Benchmark CPU vs EdgeTPU performance
4. Update integration code to use EdgeTPU model

### Long-term (Production):
1. Add EdgeTPU health monitoring
2. Fallback to CPU if EdgeTPU unavailable
3. Document model update pipeline
4. Setup automated recompilation for model updates

---

## Code Integration Strategy

The classifier already supports EdgeTPU via `use_edgetpu` parameter:

```python
# classifier.py line 20
classifier = SpeciesClassifier(
    model_path,
    labels_path,
    use_edgetpu=True  # ← Enable when EdgeTPU model ready
)
```

**Current:** `use_edgetpu=False` (CPU inference)
**Target:** `use_edgetpu=True` (EdgeTPU inference)

**Detection Pattern:**
```python
# Auto-detect EdgeTPU model
if Path("data/models/birds_v1_edgetpu.tflite").exists():
    model = "data/models/birds_v1_edgetpu.tflite"
    use_edgetpu = True
else:
    model = "data/models/birds_v1.tflite"
    use_edgetpu = False
```

---

## Testing Checklist

Once EdgeTPU model compiled:

- [ ] Verify model file exists: `birds_v1_edgetpu.tflite`
- [ ] Check file size (should be similar to original)
- [ ] Test inference with pycoral
- [ ] Compare CPU vs EdgeTPU accuracy (should be identical)
- [ ] Benchmark inference time (expect ~10x speedup)
- [ ] Verify Coral USB not overloaded (shared with Frigate)
- [ ] Test sustained operation (thermal throttling?)

---

## Performance Targets

**CPU Baseline:**
- Inference time: ~150ms per image
- Throughput: ~6 images/second
- CPU usage: ~40% single core

**EdgeTPU Target:**
- Inference time: <20ms per image
- Throughput: >50 images/second
- CPU usage: ~5% (offloaded to TPU)
- Coral USB shared with Frigate (should handle both)

**Production Load:**
- Bird visits: ~10-20 per hour
- Classifications: ~1 per minute average
- EdgeTPU utilization: <5% (plenty of headroom)

---

## Resources

- [EdgeTPU Compiler Docs](https://coral.ai/docs/edgetpu/compiler/)
- [Model Compatibility](https://coral.ai/docs/edgetpu/models-intro/)
- [Google Coral Docker Images](https://hub.docker.com/u/google-coral)
- [pycoral API Documentation](https://coral.ai/docs/reference/py/)

---

**Next Action:** Try Docker compilation (Option 1)
**Owner:** System Administrator
**Blocked By:** Need to test Docker approach
**Blocking:** Production classification deployment

---

**Last Updated:** 2025-11-14
**Status:** ACTIVE INVESTIGATION
