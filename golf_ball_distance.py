#!/usr/bin/env python3
"""
Golf Ball Distance Estimator
=============================
Estimates the distance from an RPi Camera v2 to a golf ball in an image
using the pinhole camera model and known ball diameter.

Camera:  Raspberry Pi Camera Module v2 (Sony IMX219)
         - Resolution: 3280 x 2464 (landscape) → 2464 x 3280 (portrait)
         - FOV: 62.2° (H) x 48.8° (V) landscape
         - In portrait mode: 48.8° (H) x 62.2° (V)
         - Sensor: 3.674 mm x 2.760 mm

Ball:    Golf ball diameter = 42.67 mm (1.68 inches, USGA minimum)

Tilt:    Camera is tilted back 15° from horizontal (slant angle).
         This means the optical axis points 15° above the horizon.

Geometry (Pinhole Model):
    focal_length_px = (image_width_px / 2) / tan(FOV_horizontal / 2)

    apparent_diameter_angle = 2 * arctan( (ball_px / 2) / focal_length_px )

    straight_line_distance = real_diameter / (2 * tan(apparent_diameter_angle / 2))

    Then correct for the 15° tilt to get ground distance:
        ground_distance = straight_line_distance * cos(tilt_angle)
        height          = straight_line_distance * sin(tilt_angle)
"""

import sys
import math

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────

# Golf ball
BALL_DIAMETER_MM = 42.67  # USGA regulation minimum diameter

# RPi Camera v2 specs (landscape orientation)
LANDSCAPE_RES = (3280, 2464)       # width x height
LANDSCAPE_FOV_H_DEG = 62.2         # horizontal FOV in landscape
LANDSCAPE_FOV_V_DEG = 48.8         # vertical   FOV in landscape

# Sensor physical size (Sony IMX219)
SENSOR_W_MM = 3.674
SENSOR_H_MM = 2.760

# Camera tilt (degrees above horizon)
CAMERA_TILT_DEG = 15.0


# ──────────────────────────────────────────────────────────────────────
# PORTRAIT MODE: swap horizontal/vertical
# ──────────────────────────────────────────────────────────────────────
PORTRAIT_RES = (LANDSCAPE_RES[1], LANDSCAPE_RES[0])  # 2464 x 3280
PORTRAIT_FOV_H_DEG = LANDSCAPE_FOV_V_DEG  # 48.8°
PORTRAIT_FOV_V_DEG = LANDSCAPE_FOV_H_DEG  # 62.2°


def compute_focal_length_px(image_width_px: int,
                             fov_h_deg: float) -> float:
    """
    Compute the focal length in pixels from the horizontal FOV and image width.
        f_px = (W / 2) / tan(FOV_h / 2)
    """
    fov_h_rad = math.radians(fov_h_deg)
    return (image_width_px / 2.0) / math.tan(fov_h_rad / 2.0)


def estimate_distance(ball_diameter_px: float,
                       image_width_px: int,
                       fov_h_deg: float,
                       tilt_deg: float = CAMERA_TILT_DEG,
                       ball_diameter_mm: float = BALL_DIAMETER_MM):
    """
    Estimate straight-line distance, ground distance, and height
    to a golf ball given its apparent pixel diameter.

    Returns dict with keys:
        focal_length_px, angle_subtended_deg,
        straight_line_m, ground_distance_m, height_m
    """
    f_px = compute_focal_length_px(image_width_px, fov_h_deg)

    # Angle subtended by the ball (radians)
    angle_rad = 2.0 * math.atan((ball_diameter_px / 2.0) / f_px)
    angle_deg = math.degrees(angle_rad)

    # Straight-line (along optical axis) distance in mm then metres
    straight_mm = ball_diameter_mm / (2.0 * math.tan(angle_rad / 2.0))
    straight_m = straight_mm / 1000.0

    # Correct for camera tilt to get ground-plane distance & height
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


# ──────────────────────────────────────────────────────────────────────
# BALL DETECTION (simple Hough-circles on grayscale)
# ──────────────────────────────────────────────────────────────────────

def detect_golf_ball(img_bgr: np.ndarray):
    """
    Attempt to find a golf ball via HoughCircles.
    Returns list of (x, y, radius) or empty list.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Try multiple param2 thresholds (strictest first)
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
            results = [(int(c[0]), int(c[1]), int(c[2]))
                       for c in circles[0]]
            return results
    return []


# ──────────────────────────────────────────────────────────────────────
# GUI APPLICATION
# ──────────────────────────────────────────────────────────────────────

class GolfDistanceApp:
    """Tkinter GUI for importing an image and estimating golf-ball distance."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Golf Ball Distance Estimator")
        self.root.geometry("1100x820")
        self.root.configure(bg="#1e1e1e")

        self.img_bgr = None       # original OpenCV image
        self.img_display = None   # PIL image for canvas
        self.detected_balls = []  # list of (x, y, r)
        self.selected_ball = None
        self.manual_circle = None # (x, y, r) from manual click-drag

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 11))
        style.configure("TLabel", font=("Helvetica", 11),
                        background="#1e1e1e", foreground="white")
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"),
                        background="#1e1e1e", foreground="#00ccff")

        # ── Top controls ──
        ctrl = ttk.Frame(root)
        ctrl.pack(fill=tk.X, padx=10, pady=6)

        ttk.Button(ctrl, text="📂 Import Image",
                   command=self.import_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="🔍 Auto Detect Ball",
                   command=self.auto_detect).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="📐 Manual Select (click+drag on ball)",
                   command=self.enable_manual).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="📏 Estimate Distance",
                   command=self.run_estimation).pack(side=tk.LEFT, padx=4)

        # ── Tilt angle entry ──
        ttk.Label(ctrl, text="  Tilt (°):").pack(side=tk.LEFT)
        self.tilt_var = tk.StringVar(value=str(CAMERA_TILT_DEG))
        ttk.Entry(ctrl, textvariable=self.tilt_var,
                  width=5).pack(side=tk.LEFT, padx=2)

        # ── Ball diameter override ──
        ttk.Label(ctrl, text="  Ball ⌀ (mm):").pack(side=tk.LEFT)
        self.diam_var = tk.StringVar(value=str(BALL_DIAMETER_MM))
        ttk.Entry(ctrl, textvariable=self.diam_var,
                  width=7).pack(side=tk.LEFT, padx=2)

        # ── Image canvas ──
        self.canvas = tk.Canvas(root, bg="#2b2b2b", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # ── Results panel ──
        self.result_label = ttk.Label(root, text="", style="Header.TLabel",
                                       anchor="w", wraplength=1060)
        self.result_label.pack(fill=tk.X, padx=12, pady=(0, 10))

        # Manual selection state
        self._manual_mode = False
        self._drag_start = None
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    # ── Image import ──────────────────────────────────────────────
    def import_image(self):
        path = filedialog.askopenfilename(
            title="Select golf ball image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                       ("All files", "*.*")])
        if not path:
            return
        self.img_bgr = cv2.imread(path)
        if self.img_bgr is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return
        self.detected_balls = []
        self.selected_ball = None
        self.manual_circle = None
        self._show_image()
        h, w = self.img_bgr.shape[:2]
        self.result_label.config(
            text=f"Loaded {w}×{h} image.  "
                 f"{'Portrait' if h > w else 'Landscape'} orientation detected.")

    # ── Display helpers ───────────────────────────────────────────
    def _show_image(self, overlay_bgr=None):
        img = overlay_bgr if overlay_bgr is not None else self.img_bgr.copy()
        # Fit to canvas
        cw = self.canvas.winfo_width() or 1060
        ch = self.canvas.winfo_height() or 640
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        self._scale = scale
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.img_display = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_display)

    # ── Auto detect ───────────────────────────────────────────────
    def auto_detect(self):
        if self.img_bgr is None:
            messagebox.showwarning("No image", "Import an image first.")
            return
        self.detected_balls = detect_golf_ball(self.img_bgr)
        if not self.detected_balls:
            messagebox.showinfo("Detection",
                                "No ball found. Use Manual Select instead.")
            return
        overlay = self.img_bgr.copy()
        for i, (cx, cy, r) in enumerate(self.detected_balls):
            color = (0, 255, 0) if i == 0 else (255, 200, 0)
            cv2.circle(overlay, (cx, cy), r, color, 3)
            cv2.putText(overlay, f"#{i+1} r={r}px",
                        (cx - r, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        self.selected_ball = self.detected_balls[0]
        self._show_image(overlay)
        x, y, r = self.selected_ball
        self.result_label.config(
            text=f"Detected {len(self.detected_balls)} circle(s). "
                 f"Selected #1: center=({x},{y}), "
                 f"diameter={2*r}px. Click 'Estimate Distance'.")

    # ── Manual selection ──────────────────────────────────────────
    def enable_manual(self):
        if self.img_bgr is None:
            messagebox.showwarning("No image", "Import an image first.")
            return
        self._manual_mode = True
        self.result_label.config(
            text="Manual mode: click the center of the ball and drag "
                 "to its edge, then release.")

    def _on_press(self, event):
        if self._manual_mode:
            self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._manual_mode and self._drag_start:
            self._show_image()
            sx, sy = self._drag_start
            r = int(math.hypot(event.x - sx, event.y - sy))
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                    outline="lime", width=2)

    def _on_release(self, event):
        if self._manual_mode and self._drag_start:
            sx, sy = self._drag_start
            r_display = math.hypot(event.x - sx, event.y - sy)
            # Convert back to original image coordinates
            scale = self._scale
            ox = int(sx / scale)
            oy = int(sy / scale)
            orad = int(r_display / scale)
            if orad < 2:
                return
            self.manual_circle = (ox, oy, orad)
            self.selected_ball = self.manual_circle
            self._manual_mode = False
            self._drag_start = None
            # Draw it
            overlay = self.img_bgr.copy()
            cv2.circle(overlay, (ox, oy), orad, (0, 255, 0), 3)
            self._show_image(overlay)
            self.result_label.config(
                text=f"Manual selection: center=({ox},{oy}), "
                     f"diameter={2*orad}px. Click 'Estimate Distance'.")

    # ── Distance estimation ───────────────────────────────────────
    def run_estimation(self):
        if self.img_bgr is None:
            messagebox.showwarning("No image", "Import an image first.")
            return
        if self.selected_ball is None:
            messagebox.showwarning("No ball",
                                   "Detect or manually select a ball first.")
            return

        h, w = self.img_bgr.shape[:2]
        portrait = h > w

        if portrait:
            fov_h = PORTRAIT_FOV_H_DEG   # 48.8°
            img_w = w
        else:
            fov_h = LANDSCAPE_FOV_H_DEG   # 62.2°
            img_w = w

        cx, cy, r = self.selected_ball
        ball_px = 2 * r

        try:
            tilt = float(self.tilt_var.get())
        except ValueError:
            tilt = CAMERA_TILT_DEG

        try:
            diam_mm = float(self.diam_var.get())
        except ValueError:
            diam_mm = BALL_DIAMETER_MM

        result = estimate_distance(
            ball_diameter_px=ball_px,
            image_width_px=img_w,
            fov_h_deg=fov_h,
            tilt_deg=tilt,
            ball_diameter_mm=diam_mm,
        )

        # Convert to yards and feet
        ground_m = result["ground_distance_m"]
        straight_m = result["straight_line_m"]
        ground_yd = ground_m * 1.09361
        ground_ft = ground_m * 3.28084

        text = (
            f"Ball diameter: {ball_px} px  |  "
            f"Focal length: {result['focal_length_px']:.1f} px  |  "
            f"Subtended angle: {result['angle_subtended_deg']:.4f}°\n"
            f"Straight-line distance: {straight_m:.2f} m  |  "
            f"Ground distance: {ground_m:.2f} m  "
            f"({ground_ft:.1f} ft / {ground_yd:.1f} yd)  |  "
            f"Camera height component: {result['height_m']:.2f} m  |  "
            f"Tilt: {tilt}°"
        )
        self.result_label.config(text=text)

        # Annotate image
        overlay = self.img_bgr.copy()
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 3)
        label = f"{ground_m:.2f}m ({ground_yd:.1f}yd)"
        cv2.putText(overlay, label, (cx - r, cy - r - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        self._show_image(overlay)


# ──────────────────────────────────────────────────────────────────────
# COMMAND-LINE (HEADLESS) MODE
# ──────────────────────────────────────────────────────────────────────

def cli_mode(image_path: str):
    """Run distance estimation without GUI."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot read '{image_path}'")
        sys.exit(1)

    h, w = img.shape[:2]
    portrait = h > w
    fov_h = PORTRAIT_FOV_H_DEG if portrait else LANDSCAPE_FOV_H_DEG
    orientation = "Portrait" if portrait else "Landscape"
    print(f"Image: {w}×{h} ({orientation})")
    print(f"Using FOV_H = {fov_h}°, tilt = {CAMERA_TILT_DEG}°\n")

    balls = detect_golf_ball(img)
    if not balls:
        print("No ball detected. Please use the GUI for manual selection.")
        sys.exit(0)

    for i, (cx, cy, r) in enumerate(balls):
        ball_px = 2 * r
        res = estimate_distance(ball_px, w, fov_h)
        gm = res["ground_distance_m"]
        print(f"Ball #{i+1}: center=({cx},{cy}), diameter={ball_px}px")
        print(f"  Focal length : {res['focal_length_px']:.1f} px")
        print(f"  Subtended    : {res['angle_subtended_deg']:.4f}°")
        print(f"  Straight-line: {res['straight_line_m']:.2f} m")
        print(f"  Ground dist  : {gm:.2f} m  "
              f"({gm*3.28084:.1f} ft / {gm*1.09361:.1f} yd)")
        print(f"  Height comp  : {res['height_m']:.2f} m")
        print()


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode: python golf_distance.py <image_path>
        cli_mode(sys.argv[1])
    else:
        # GUI mode
        root = tk.Tk()
        app = GolfDistanceApp(root)
        root.mainloop()
