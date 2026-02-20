Here's the complete Python application: golf_ball_distance.py

## How It Works

The app uses the **pinhole camera model** to estimate distance from the apparent pixel size of a golf ball in the image. The core formula is: [blog.sectorr](https://blog.sectorr.dev/Camera-Distance/)

\[
f_{px} = \frac{W_{px} / 2}{\tan(\text{FOV}_h / 2)}
\]

where \( W_{px} \) is image width in pixels and \( \text{FOV}_h \) is the horizontal field of view. The straight-line distance is then: [scratchapixel](https://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/how-pinhole-camera-works-part-2.html)

\[
D = \frac{d_{real}}{2 \cdot \tan\!\left(\frac{1}{2} \cdot 2\arctan\!\left(\frac{d_{px}/2}{f_{px}}\right)\right)}
\]

which simplifies to \( D = d_{real} \cdot f_{px} / d_{px} \), and the 15° tilt correction projects onto the ground plane via \( D_{ground} = D \cdot \cos(15°) \). [reddit](https://www.reddit.com/r/computervision/comments/1cw612t/how_to_identify_distance_from_the_camera_to_an/)

## Camera Configuration

| Parameter | Value |
|---|---|
| Sensor | Sony IMX219 (RPi Camera v2)  [sparkfun](https://www.sparkfun.com/raspberry-pi-camera-module-v2.html) |
| Resolution | 3280 × 2464 (landscape) → 2464 × 3280 (portrait)  [opensourceinstruments](https://www.opensourceinstruments.com/Electronics/Data/UC350D.pdf) |
| Landscape FOV | 62.2° × 48.8° |
| Portrait FOV | 48.8° (H) × 62.2° (V) — axes swap |
| Camera tilt | 15° above horizon (adjustable in UI) |
| Golf ball ⌀ | 42.67 mm (USGA regulation) |

## Features

- **Image import** via file dialog (JPG, PNG, BMP, TIFF)
- **Auto-detection** using OpenCV HoughCircles with cascading sensitivity thresholds
- **Manual selection** — click the ball center and drag to its edge for cases where auto-detect fails
- **Tilt & diameter overrides** — adjustable in the toolbar so you can experiment with different mounting angles
- **Dual mode** — run with GUI (`python golf_ball_distance.py`) or headless CLI (`python golf_ball_distance.py image.jpg`)
- **Output** in meters, feet, and yards, plus the subtended angle and height component

## Key Design Decisions

- **Portrait mode auto-detection**: the app checks if `height > width` and swaps the FOV axes accordingly, since your spec says portrait orientation. [reddit](https://www.reddit.com/r/raspberry_pi/comments/yntdla/understanding_raspi_cam_v2_resolution_vs_fov/)
- **Tilt correction**: the camera's 15° backward tilt means the optical axis isn't parallel to the ground. The straight-line distance along the optical axis is decomposed into a ground-plane component (\(\cos 15°\)) and a vertical component (\(\sin 15°\)). [reddit](https://www.reddit.com/r/computervision/comments/1cw612t/how_to_identify_distance_from_the_camera_to_an/)
- **HoughCircles cascade**: tries `param2` values of 50→40→30→20, stopping at the first successful detection to balance precision and recall.

## Running the App

```bash
# GUI mode (opens file browser)
python golf_ball_distance.py

# CLI mode (auto-detect + print results)
python golf_ball_distance.py path/to/golf_image.jpg
```

**Dependencies**: `opencv-python`, `numpy`, `Pillow`, and `tkinter` (included with most Python distributions).                                         

