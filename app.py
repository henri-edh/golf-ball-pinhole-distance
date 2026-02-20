#!/usr/bin/env python3
import os
import math
import tempfile

from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np

# ─────────────────────────────
# CONSTANTS / CAMERA GEOMETRY
# ─────────────────────────────

BALL_DIAMETER_MM = 42.67  # golf ball diameter

# RPi Camera v2 landscape specs (62.2 x 48.8 deg)
LANDSCAPE_RES = (3280, 2464)         # width x height (for reference only)
LANDSCAPE_FOV_H_DEG = 62.2
LANDSCAPE_FOV_V_DEG = 48.8

# Portrait = rotated 90°, FOV axes swap
PORTRAIT_FOV_H_DEG = LANDSCAPE_FOV_V_DEG  # 48.8
PORTRAIT_FOV_V_DEG = LANDSCAPE_FOV_H_DEG  # 62.2

DEFAULT_TILT_DEG = 15.0  # camera tilted back 15°


def compute_focal_length_px(image_width_px: int, fov_h_deg: float) -> float:
    """f_px = (W/2) / tan(FOV_h/2)."""
    fov_h_rad = math.radians(fov_h_deg)
    return (image_width_px / 2.0) / math.tan(fov_h_rad / 2.0)


def estimate_distance(ball_diameter_px: float,
                      image_width_px: int,
                      fov_h_deg: float,
                      tilt_deg: float = DEFAULT_TILT_DEG,
                      ball_diameter_mm: float = BALL_DIAMETER_MM):
    """Return distance metrics for the ball."""
    f_px = compute_focal_length_px(image_width_px, fov_h_deg)

    angle_rad = 2.0 * math.atan((ball_diameter_px / 2.0) / f_px)
    angle_deg = math.degrees(angle_rad)

    straight_mm = ball_diameter_mm / (2.0 * math.tan(angle_rad / 2.0))
    straight_m = straight_mm / 1000.0

    tilt_rad = math.radians(tilt_deg)
    ground_m = straight_m * math.cos(tilt_rad)
    height_m = straight_m * math.sin(tilt_rad)

    return {
        "focal_length_px": f_px,
        "angle_subtended_deg": angle_deg,
        "straight_line_m": straight_m,
        "ground_distance_m": ground_m,
        "height_m": height_m,
    }


def detect_golf_ball(img_bgr):
    """Simple Hough circle detection; returns list of (x, y, r)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    for param2 in [50, 40, 30, 20]:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=param2,
            minRadius=5,
            maxRadius=500,
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]
    return []


# ─────────────────────────────
# FLASK APP
# ─────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(tempfile.gettempdir(), "golf_uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        tilt_str = (request.form.get("tilt_deg") or "").strip()
        diam_str = (request.form.get("ball_diam_mm") or "").strip()

        if not file or file.filename == "":
            error = "Please choose an image file."
            return render_template("index.html",
                                   result=result,
                                   error=error,
                                   default_tilt=DEFAULT_TILT_DEG,
                                   default_diam=BALL_DIAMETER_MM)

        # Robust tilt parsing, default 15°
        try:
            tilt_deg = float(tilt_str) if tilt_str else DEFAULT_TILT_DEG
        except ValueError:
            tilt_deg = DEFAULT_TILT_DEG

        # Robust diameter parsing, default 42.67 mm
        try:
            ball_diam_mm = float(diam_str) if diam_str else BALL_DIAMETER_MM
        except ValueError:
            ball_diam_mm = BALL_DIAMETER_MM

        in_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(in_path)

        img = cv2.imread(in_path)
        if img is None:
            error = "Could not read the uploaded image."
            return render_template("index.html",
                                   result=result,
                                   error=error,
                                   default_tilt=DEFAULT_TILT_DEG,
                                   default_diam=BALL_DIAMETER_MM)

        h, w = img.shape[:2]
        portrait = h > w
        fov_h = PORTRAIT_FOV_H_DEG if portrait else LANDSCAPE_FOV_H_DEG

        circles = detect_golf_ball(img)
        if not circles:
            error = "No circles detected. Try a clearer ball image."
            return render_template("index.html",
                                   result=result,
                                   error=error,
                                   default_tilt=DEFAULT_TILT_DEG,
                                   default_diam=BALL_DIAMETER_MM)

        cx, cy, r = circles[0]
        ball_px = 2 * r

        dist = estimate_distance(
            ball_diameter_px=ball_px,
            image_width_px=w,
            fov_h_deg=fov_h,
            tilt_deg=tilt_deg,
            ball_diameter_mm=ball_diam_mm,
        )

        ground_m = dist["ground_distance_m"]
        straight_m = dist["straight_line_m"]
        ground_ft = ground_m * 3.28084
        ground_yd = ground_m * 1.09361

        # Annotate image
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 3)
        label = f"{ground_m:.2f} m ({ground_yd:.1f} yd)"
        cv2.putText(
            overlay, label, (max(cx - r, 10), max(cy - r - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        annotated_filename = f"annotated_{file.filename}"
        out_path = os.path.join(app.config["UPLOAD_FOLDER"], annotated_filename)
        cv2.imwrite(out_path, overlay)

        result = {
            "orientation": "Portrait" if portrait else "Landscape",
            "image_width": w,
            "image_height": h,
            "tilt_deg": tilt_deg,
            "ball_diam_mm": ball_diam_mm,
            "ball_px": ball_px,
            "focal_length_px": dist["focal_length_px"],
            "angle_deg": dist["angle_subtended_deg"],
            "straight_m": straight_m,
            "ground_m": ground_m,
            "ground_ft": ground_ft,
            "ground_yd": ground_yd,
            "height_m": dist["height_m"],
            "annotated_image": annotated_filename,
        }

    return render_template("index.html",
                           result=result,
                           error=error,
                           default_tilt=DEFAULT_TILT_DEG,
                           default_diam=BALL_DIAMETER_MM)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # Dev server: http://127.0.0.1:5000
    app.run(debug=True, host="0.0.0.0", port=5000)
