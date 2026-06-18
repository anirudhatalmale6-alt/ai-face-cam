# AI Face Cam

Virtual webcam tool that animates a face photo with keyboard controls.
Outputs through OBS Virtual Camera for use in video calls, KYC verification, etc.

## Setup (Windows)

1. Install OBS Studio (for virtual camera driver): https://obsproject.com/
2. Start OBS once, enable "Start Virtual Camera" (this installs the driver)
3. Run: `python face_cam.py -i your_face.png`

## Controls

| Key | Action |
|-----|--------|
| A/D | Turn head left/right |
| W/S | Look up/down |
| Q/E | Tilt head |
| B   | Blink |
| +/- | Zoom in/out |
| R   | Reset position |
| ESC | Quit |

## Usage

```
python face_cam.py -i face_photo.png
python face_cam.py -i face_photo.png --no-virtual-cam
```

Best results with a front-facing, well-lit face photo (passport-style).
