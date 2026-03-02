# Audit History — Distance Estimation Bug

## Bug Found in `app.py` line 166

**Problem:** `sensor_horiz_px = h` (image height) is paired with `PORTRAIT_FOV_SHORT_DEG` (48.8°), but 48.8° is the FOV across the image **width**, not height.

**Effect:** Focal length inflated by `h/w` ratio (4/3 for 4:3 images = 33% overestimate on all distances).

**Example from user's image (480×640 portrait):**
- Buggy:  f = 640/2 / tan(24.4°) = 705.4 px → 1.61 m straight-line → 1.60 m ground
- Correct: f = 480/2 / tan(24.4°) = 529.5 px → 1.21 m straight-line → 1.20 m ground

## Fix (not yet applied)

In `process_image()` at line 166, change:
```python
# BEFORE (buggy):
h, w = img.shape[:2]
sensor_horiz_px = h   # portrait: sensor horizontal = file height
fov_h = PORTRAIT_FOV_SHORT_DEG

# AFTER (correct):
h, w = img.shape[:2]
if w >= h:  # landscape
    sensor_horiz_px = w
    fov_h = LANDSCAPE_FOV_H_DEG
else:  # portrait
    sensor_horiz_px = w
    fov_h = PORTRAIT_FOV_SHORT_DEG
```

`sensor_horiz_px` should always be `w` (image width). FOV selected by orientation.

## `table_dist.py` — Verified Correct

- Properly pairs 1920w with 62.2° and 1080w with 48.8°
- Pinhole math is the inverse of `estimate_distance()` in `app.py`
- CSV output is valid for images at those specific resolutions (1920w landscape, 1080w portrait)

## Key Insight: Resolution Matters

The CSV pixel values are for 1080w portrait images. Comparing directly to a 480w image is invalid — must scale by 1080/480 = 2.25× first. 18.7 px at 480w = 42.1 px equivalent at 1080w (closer than 1.4 m, off the CSV range).

## FOV Constants (62.2° H, 48.8° V)

These are consistent with a **4:3 native sensor** (typical phone camera), not 16:9.
