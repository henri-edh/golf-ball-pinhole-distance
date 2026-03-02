import math
import pandas as pd

# ─────────────────────────────
# CAMERA + BALL PARAMETERS
# ─────────────────────────────

BALL_DIAMETER_MM = 42.67       # golf ball diameter
CAMERA_HEIGHT_M  = 0.250       # lens height above ground (250 mm)

# Lens FOV on the full binned sensor (1640 x 1232)
FOV_H_DEG = 62.2               # horizontal FOV
FOV_V_DEG = 48.8               # vertical FOV

# Effective sensor in 2x2 binned mode
SENSOR_BINNED_W = 1640         # full-FOV width (used for calibration)
SENSOR_BINNED_H = 1232         # full-FOV height (not used here)

# Fusion stream: 640 x 480 crop taken from the 1640 x 1232 frame.
# IMPORTANT: cropping does NOT change focal length; we still calibrate
# using the full 1640px width and the 62.2° FOV.

# ─────────────────────────────
# FOCAL LENGTH (in pixels)
# ─────────────────────────────

def focal_length_px(sensor_horiz_px: int, fov_h_deg: float) -> float:
    """Compute focal length in pixels from sensor width and horizontal FOV."""
    return (sensor_horiz_px / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)

F_PX = focal_length_px(SENSOR_BINNED_W, FOV_H_DEG)


# ─────────────────────────────
# BALL SIZE IN PIXELS VS DISTANCE
# ─────────────────────────────

def ball_pixel_diameter(ground_m: float,
                        f_px: float = F_PX,
                        camera_height_m: float = CAMERA_HEIGHT_M,
                        ball_mm: float = BALL_DIAMETER_MM) -> float:
    """
    Expected pixel diameter of the ball on the 640x480 crop, given
    ground distance in metres.

    We use the pinhole model with:
      straight_m = sqrt(ground_m^2 + camera_height_m^2)
      angle_rad  = 2 * atan( (ball_mm/2) / (straight_m * 1000) )
      px_diam    = 2 * f_px * tan(angle_rad / 2)
    """
    straight_m = math.sqrt(ground_m**2 + camera_height_m**2)
    angle_rad  = 2.0 * math.atan((ball_mm / 2.0) / (straight_m * 1000.0))
    px_diam    = 2.0 * f_px * math.tan(angle_rad / 2.0)
    return px_diam, straight_m


# ─────────────────────────────
# BUILD TABLE 1.4m → 3.2m (0.1m steps)
# ─────────────────────────────

rows = []

for d_mm in range(1400, 3300, 100):   # 1400mm to 3200mm inclusive
    ground_m = d_mm / 1000.0
    px_diam, straight_m = ball_pixel_diameter(ground_m)

    rows.append({
        "Distance_ground_m":  round(ground_m, 1),
        "Distance_straight_m": round(straight_m, 3),
        "Ball_pixels_width":  round(px_diam, 2),
    })

df = pd.DataFrame(rows)

# ─────────────────────────────
# OUTPUT
# ─────────────────────────────

print(df.to_string(index=False))

# Save to CSV for later use in the app / analysis
df.to_csv("golf_ball_pixels_1640bin_640crop.csv", index=False)
print("\nSaved to golf_ball_pixels_1640bin_640crop.csv")
