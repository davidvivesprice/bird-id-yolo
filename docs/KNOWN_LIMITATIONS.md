# Known Limitations & Future Improvements

## Current Challenge: Feeder Swaying

### Problem Statement
The bird feeder moves significantly due to:
- Wind
- Birds landing/taking off
- General outdoor conditions

**Current motion detection cannot distinguish between:**
- Feeder swaying (false positive - should ignore)
- Bird on feeder (true positive - should detect)

### Why MOG2 Won't Solve This
MOG2 is excellent for:
- ✅ Adapting to lighting changes
- ✅ Filtering shadows
- ✅ Learning static backgrounds

But it **cannot** help with feeder sway because:
- ❌ The feeder IS actually moving (it's real motion, not noise)
- ❌ Both sway and bird activity are foreground motion
- ❌ No way to distinguish without understanding motion patterns

### Test Case
**File:** `data/clips/baseline/feeder swaying.mp4`
- Shows feeder movement without bird activity
- Should produce 0% motion detection (ideal)
- Currently produces false positives

### Solutions (Phase 2 - Post-MOG2)

#### Option 1: Motion Stabilization (Recommended)
**How it works:**
1. Digitally stabilize video using feature tracking
2. Compensate for camera/feeder movement
3. Run motion detection on stabilized frames
4. Only new motion (birds) detected

**Libraries:**
- OpenCV `estimateRigidTransform()` / `findTransformECC()`
- VidStab library

**Pros:**
- Directly addresses the problem
- Works with existing detection
- Proven technique

**Cons:**
- Adds processing overhead
- May crop edges of frame
- Needs tuning per setup

#### Option 2: Optical Flow Analysis
**How it works:**
1. Calculate motion vectors for all pixels
2. Uniform/coherent vectors = whole scene moving (feeder sway)
3. Isolated/chaotic vectors = localized motion (bird)

**Libraries:**
- OpenCV `calcOpticalFlowFarneback()`
- Lucas-Kanade method

**Pros:**
- Can distinguish motion types
- No frame cropping

**Cons:**
- More complex to implement
- Computationally expensive
- Requires careful threshold tuning

#### Option 3: Temporal Pattern Analysis
**How it works:**
1. Track motion patterns over time
2. Rhythmic/periodic = wind-driven sway
3. Sudden/erratic = bird activity

**Approach:**
- Frequency analysis of motion
- Learn feeder sway "signature"
- Filter out matching patterns

**Pros:**
- Smart filtering
- Can work alongside other methods

**Cons:**
- Requires training data
- Complex implementation
- May miss birds during windy periods

#### Option 4: Machine Learning Classifier
**How it works:**
1. Train model on labeled clips:
   - "Sway only" clips
   - "Bird activity" clips
2. Classify detected motion regions

**Pros:**
- Can learn complex patterns
- Highly accurate when trained well

**Cons:**
- Needs labeled training data
- ML infrastructure required
- Overkill for this specific problem

### Recommended Approach

**Phase 1 (Current):**
- ✅ Implement config + logging system
- ⏳ Implement MOG2 (improves lighting/shadows)
- 📝 Document feeder sway as known limitation
- 🔧 Tune thresholds to minimize false positives

**Phase 2 (After MOG2):**
1. Collect representative clips:
   - Feeder swaying (no birds)
   - Birds with feeder movement
   - Various wind conditions
2. Implement **Motion Stabilization** (Option 1)
3. Test on collected clips
4. Tune stabilization parameters
5. Deploy if results improve

**Phase 3 (If needed):**
- Add Optical Flow if stabilization insufficient
- Combine multiple approaches if necessary

### Workarounds (Temporary)

**1. Increase thresholds:**
```yaml
# config.yaml
motion:
  diff_threshold: 25    # Higher = less sensitive
  min_area: 1000        # Larger = ignores small sway
```

**2. Add consecutive frame requirement:**
```yaml
motion:
  min_consecutive_frames: 5  # Motion must persist 5 frames
```
- Feeder sway might be sporadic
- Bird activity tends to persist longer

**3. Manual review:**
- Accept some false positives
- Filter during post-processing
- Human review of flagged events

### Related Files

**Test clips:**
- `data/clips/baseline/feeder swaying.mp4` - Pure sway (should be ignored)
- `data/clips/cardinal/Birds*.mp4` - Bird + potential sway

**Documentation:**
- This file - Known limitations
- `docs/IMPLEMENTATION_LOG.md` - Current implementation status
- `TEST_RESULTS.md` - Test results and metrics

---

## Other Limitations

### Lighting Conditions
**Issue:** Extreme lighting changes (sunrise, direct sun, shadows)  
**Solution:** ✅ MOG2 implementation (Priority 3)  
**Status:** Planned

### Small Birds
**Issue:** Very small birds might be filtered by min_area threshold  
**Solution:** Tune min_area dynamically or use multi-scale detection  
**Status:** Monitor during testing

### Overlapping Motion
**Issue:** Multiple birds = merged bounding boxes  
**Solution:** Better contour separation or instance segmentation  
**Status:** Low priority - acceptable for now

---

**Last Updated:** 2025-11-13  
**Next Review:** After MOG2 implementation
