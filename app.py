#!/usr/bin/env python3
import os
import math
import tempfile
import logging

os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "yolo_cfg")

from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ─────────────────────────────
# YOLO MODEL
# ─────────────────────────────

_yolo_model = YOLO("yolo11n.pt")
SPORTS_BALL_CLASS = 32

# ─────────────────────────────
# CONSTANTS / CAMERA GEOMETRY
# ─────────────────────────────

BALL_DIAMETER_MM         = 42.67
CAMERA_HEIGHT_M          = 0.250
DEFAULT_CAMERA_HEIGHT_MM = 250

# Lens FOV — these are fixed properties of the lens, not the resolution
FOV_H_DEG = 62.2    # horizontal FOV of the full sensor
FOV_V_DEG = 48.8    # vertical FOV of the full sensor

# ── Sensor modes ──────────────────────────────────────────────────────
#
# Mode 1: 2×2 binned full-FOV frame (stills / reference images)
#   Resolution : 1640 × 1232
#   FOV        : full 62.2° H × 48.8° V
#
# Mode 2: 640 × 480 centre-crop @ 180 fps (fusion / tracking stream)
#   Resolution : 640 × 480 — a crop out of the 1640 × 1232 binned frame
#   FOV        : REDUCED (only the centre portion of the sensor is used)
#   f_px       : same physical focal length → same f_px as Mode 1
#                because f_px is calibrated from the full 1640px width.
#
# The ball pixel diameter measured in Mode 2 is fed into the pinhole
# formula using F_PX derived from the full 1640px sensor.
# ──────────────────────────────────────────────────────────────────────
SENSOR_BINNED_W  = 1024    # full-FOV binned frame width  (Mode 1)
SENSOR_BINNED_H  = 768    # full-FOV binned frame height (Mode 1)

CROP_W           = 1024     # fusion stream width  (Mode 2)
CROP_H           = 768     # fusion stream height (Mode 2)
# SENSOR_BINNED_W  = 1640    # full-FOV binned frame width  (Mode 1)
# SENSOR_BINNED_H  = 1232    # full-FOV binned frame height (Mode 1)

#CROP_W           = 640     # fusion stream width  (Mode 2)
# CROP_H           = 480     # fusion stream height (Mode 2)

# Cropped FOV — what fraction of the full sensor the crop covers
# Used only for display / reference; distance calc uses F_PX directly.
CROP_FOV_H_DEG = 2.0 * math.degrees(
    math.atan((CROP_W / SENSOR_BINNED_W) * math.tan(math.radians(FOV_H_DEG) / 2.0))
)
CROP_FOV_V_DEG = 2.0 * math.degrees(
    math.atan((CROP_H / SENSOR_BINNED_H) * math.tan(math.radians(FOV_V_DEG) / 2.0))
)

# ── Single focal length for ALL distance calculations ─────────────────
# Calibrated from the full 1640px-wide binned sensor and the lens FOV.
# Cropping to 640×480 does NOT change this value.
F_PX = (SENSOR_BINNED_W / 2.0) / math.tan(math.radians(FOV_H_DEG) / 2.0)


def estimate_distance(ball_diameter_px: float,
                      camera_height_m: float = CAMERA_HEIGHT_M,
                      ball_diameter_mm: float = BALL_DIAMETER_MM) -> dict:
    """
    Pinhole model + Pythagorean ground distance.
    ball_diameter_px may come from either the 1640-wide frame or the
    640-wide crop — F_PX is the same for both (same lens, same pixel pitch).

      straight_m = lens-to-ball distance from angular size
      ground_m   = sqrt(straight_m² - camera_height_m²)
    """
    angle_rad  = 2.0 * math.atan((ball_diameter_px / 2.0) / F_PX)
    straight_m = (ball_diameter_mm / (2.0 * math.tan(angle_rad / 2.0))) / 1000.0
    ground_m   = math.sqrt(max(straight_m ** 2 - camera_height_m ** 2, 0.0))

    return {
        "focal_length_px":     F_PX,
        "angle_subtended_deg": math.degrees(angle_rad),
        "straight_line_m":     straight_m,
        "ground_distance_m":   ground_m,
    }


def get_all_detections(img_bgr):
    """Return list of all YOLO sports ball detections as (cx, cy, r, conf)."""
    results = _yolo_model.predict(
        source=img_bgr,
        classes=[SPORTS_BALL_CLASS],
        conf=0.15,
        imgsz=640,      # matches the fusion stream native resolution
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cx   = (x1 + x2) / 2.0
        cy   = (y1 + y2) / 2.0
        r    = ((x2 - x1) + (y2 - y1)) / 4.0
        detections.append((cx, cy, r, conf))

    return detections


def pick_detection(detections, hint_x=None, hint_y=None):
    """Pick nearest to click hint, or highest confidence if no hint."""
    if not detections:
        return None
    if hint_x is not None and hint_y is not None:
        return min(detections,
                   key=lambda d: math.hypot(d[0] - hint_x, d[1] - hint_y))
    return max(detections, key=lambda d: d[3])


def draw_annotated(img, detections, chosen, camera_height_m, ball_diam_mm):
    """
    Draw all detections (grey) and highlight chosen one (green).
    Circle is drawn at MIN_DRAW_RADIUS minimum so it's always visible.
    Distance is computed from the ACTUAL detected radius, not draw radius.
    """
    overlay = img.copy()
    MIN_DRAW_RADIUS = 18

    # Non-chosen detections in grey
    for d in detections:
        if d is chosen:
            continue
        cx, cy, r, conf = d
        draw_r = max(int(r), MIN_DRAW_RADIUS)
        cv2.circle(overlay, (int(cx), int(cy)), draw_r, (120, 120, 120), 1)
        cv2.circle(overlay, (int(cx), int(cy)), 3, (120, 120, 120), -1)
        cv2.putText(overlay, f"{conf:.0%}",
                    (int(cx) + draw_r + 4, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # Chosen detection in green
    cx, cy, r, conf = chosen
    draw_r = max(int(r), MIN_DRAW_RADIUS)
    col = (0, 255, 0)

    cv2.circle(overlay, (int(cx), int(cy)), draw_r, col, 2)
    cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    # Use actual detected radius for distance, not draw_r
    dist     = estimate_distance(2.0 * r, camera_height_m, ball_diam_mm)
    ground_m = dist["ground_distance_m"]

    label = f"{ground_m:.2f}m ({ground_m * 1.09361:.1f}yd) {conf:.0%}"
    lx = max(int(cx) - draw_r, 10)
    ly = max(int(cy) - draw_r - 10, 30)
    cv2.putText(overlay, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    return overlay, dist, ground_m


# ─────────────────────────────
# FLASK APP
# ─────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(tempfile.gettempdir(), "golf_uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def save_upload(file_storage):
    filename = file_storage.filename
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(path)
    return filename, path


def process_image(img, filename, camera_height_m, ball_diam_mm,
                  hint_x=None, hint_y=None):
    """Full pipeline: detect → pick → annotate → result dict or error string."""
    h, w = img.shape[:2]

    detections = get_all_detections(img)
    if not detections:
        return "YOLO11 found no sports ball. Try clicking directly on the ball."

    chosen = pick_detection(detections, hint_x, hint_y)
    overlay, dist, ground_m = draw_annotated(
        img, detections, chosen, camera_height_m, ball_diam_mm)

    annotated_filename = f"annotated_{filename}"
    cv2.imwrite(
        os.path.join(app.config["UPLOAD_FOLDER"], annotated_filename), overlay)

    cx, cy, r, confidence = chosen
    mode = "Click" if hint_x is not None else "Auto"

    return {
        # Image info
        "image_width":      w,
        "image_height":     h,
        # Camera config
        "camera_height_mm": int(camera_height_m * 1000),
        "camera_height_m":  camera_height_m,
        "focal_length_px":  F_PX,
        "sensor_mode":      f"{SENSOR_BINNED_W}×{SENSOR_BINNED_H} binned → "
                            f"{CROP_W}×{CROP_H} crop",
        "crop_fov":         f"{CROP_FOV_H_DEG:.1f}° H × {CROP_FOV_V_DEG:.1f}° V",
        # Ball
        "ball_diam_mm":     ball_diam_mm,
        "ball_px":          round(2.0 * r, 1),
        # Distance
        "angle_deg":        dist["angle_subtended_deg"],
        "straight_m":       dist["straight_line_m"],
        "ground_m":         ground_m,
        "ground_ft":        ground_m * 3.28084,
        "ground_yd":        ground_m * 1.09361,
        # Detection meta
        "confidence":       confidence,
        "n_detections":     len(detections),
        "annotated_image":  annotated_filename,
        "orig_image":       filename,
        "mode":             mode,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result      = None
    error       = None
    image_info  = None
    default_camera_height_mm = DEFAULT_CAMERA_HEIGHT_MM
    default_diam = BALL_DIAMETER_MM

    if request.method == "POST":
        action = request.form.get("action", "upload")

        try:
            camera_height_m = float(
                (request.form.get("camera_height_mm") or "").strip()
            ) / 1000.0
        except ValueError:
            camera_height_m = CAMERA_HEIGHT_M

        try:
            ball_diam_mm = float(
                (request.form.get("ball_diam_mm") or "").strip()
            )
        except ValueError:
            ball_diam_mm = BALL_DIAMETER_MM

        default_camera_height_mm = int(camera_height_m * 1000)
        default_diam = ball_diam_mm

        # ── Upload: new image ────────────────────────────────────────
        if action == "upload":
            file = request.files.get("image")
            if not file or file.filename == "":
                error = "Please choose an image file."
                return render_template(
                    "index.html", result=None, error=error,
                    default_camera_height_mm=default_camera_height_mm,
                    default_diam=default_diam, image_info=None)

            filename, img_path = save_upload(file)
            img = cv2.imread(img_path)
            if img is None:
                error = "Could not read the uploaded image."
                return render_template(
                    "index.html", result=None, error=error,
                    default_camera_height_mm=default_camera_height_mm,
                    default_diam=default_diam, image_info=None)

            h, w = img.shape[:2]
            image_info = {"filename": filename, "width": w, "height": h}

            out = process_image(img, filename, camera_height_m, ball_diam_mm)
            if isinstance(out, str):
                error = out
            else:
                result = out

        # ── Click override: same image, new hint ─────────────────────
        elif action == "click_override":
            filename = request.form.get("current_image")
            if not filename:
                error = "No image loaded. Please upload again."
                return render_template(
                    "index.html", result=None, error=error,
                    default_camera_height_mm=default_camera_height_mm,
                    default_diam=default_diam, image_info=None)

            img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            img = cv2.imread(img_path)
            if img is None:
                error = "Could not read stored image. Please upload again."
                return render_template(
                    "index.html", result=None, error=error,
                    default_camera_height_mm=default_camera_height_mm,
                    default_diam=default_diam, image_info=None)

            h, w = img.shape[:2]
            image_info = {"filename": filename, "width": w, "height": h}

            try:
                hint_x = float(request.form.get("click_x"))
                hint_y = float(request.form.get("click_y"))
            except (TypeError, ValueError):
                hint_x = hint_y = None

            out = process_image(img, filename, camera_height_m, ball_diam_mm,
                                hint_x, hint_y)
            if isinstance(out, str):
                error = out
            else:
                result = out

        return render_template(
            "index.html", result=result, error=error,
            default_camera_height_mm=default_camera_height_mm,
            default_diam=default_diam, image_info=image_info)

    return render_template(
        "index.html", result=result, error=error,
        default_camera_height_mm=default_camera_height_mm,
        default_diam=default_diam, image_info=None)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)
