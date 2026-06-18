"""
AI Face Cam 3D - Realistic face animation for virtual webcam
- AI face generation (no photo needed)
- Perspective-based head rotation (smooth, no mesh artifacts)
- Natural expressions (blink, smile, mouth)
- Virtual webcam output via OBS
"""

import cv2
import numpy as np
import math
import time
import sys
import os
import urllib.request
import tempfile

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
FACES_DIR = os.path.join(os.path.expanduser("~"), ".aifacecam", "faces")

CAM_WIDTH = 640
CAM_HEIGHT = 480
FPS = 30


def ensure_dirs():
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def ensure_models():
    needed = [
        ("buffalo_l/det_10g.onnx",
         "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"),
        ("buffalo_l/2d106det.onnx", None),
    ]
    dest_dir = os.path.join(MODELS_DIR, "buffalo_l")
    all_exist = all(
        os.path.exists(os.path.join(dest_dir, os.path.basename(f[0])))
        for f in needed
    )
    if all_exist:
        return

    os.makedirs(dest_dir, exist_ok=True)
    zip_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    zip_path = os.path.join(MODELS_DIR, "buffalo_l.zip")

    print("Downloading face detection models (~300MB, one-time)...")
    try:
        def progress(count, block_size, total_size):
            if total_size > 0:
                pct = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r  {pct}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(zip_url, zip_path, reporthook=progress)
        print("\n  Extracting...")

        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                basename = os.path.basename(name)
                if basename in ("det_10g.onnx", "2d106det.onnx"):
                    data = zf.read(name)
                    with open(os.path.join(dest_dir, basename), 'wb') as f:
                        f.write(data)
                    print(f"  Extracted: {basename}")
        os.remove(zip_path)
    except Exception as e:
        print(f"\n  Model download failed: {e}")
        print("  Will use basic face detection")


def generate_ai_face():
    """Download a random AI-generated face from thispersondoesnotexist.com"""
    print("Generating AI face...")
    try:
        url = "https://thispersondoesnotexist.com"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response = urllib.request.urlopen(req, timeout=15)
        data = response.read()

        img_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is not None:
            save_path = os.path.join(FACES_DIR, f"face_{int(time.time())}.jpg")
            cv2.imwrite(save_path, img)
            print(f"  Generated: {img.shape[1]}x{img.shape[0]}")
            return img
    except Exception as e:
        print(f"  Generation failed: {e}")
    return None


def get_face_analysis():
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            root=os.path.dirname(MODELS_DIR),
            allowed_modules=['detection', 'landmark_2d_106'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        print(f"  InsightFace unavailable: {e}")
        return None


def detect_landmarks(image, face_app=None):
    """Detect face landmarks, return (landmarks_106, bbox) or fallback"""
    if face_app is not None:
        try:
            faces = face_app.get(image)
            if faces:
                face = faces[0]
                lm = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None else None
                bbox = face.bbox.astype(int)
                if lm is not None:
                    return lm, bbox
                print("  InsightFace: face found but no 106 landmarks, using bbox only")
                return None, bbox
        except Exception as e:
            print(f"  InsightFace detection error: {e}")

    h, w = image.shape[:2]
    try:
        xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(xml_path):
            cascade = cv2.CascadeClassifier(xml_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) > 0:
                x, y, fw, fh = faces[np.argmax([fw * fh for (x, y, fw, fh) in faces])]
                bbox = np.array([x, y, x + fw, y + fh])
                return None, bbox
    except Exception as e:
        print(f"  Haar cascade unavailable: {e}")

    m = min(w, h) // 6
    bbox = np.array([m, m, w - m, h - m])
    print(f"  Using center-crop face estimate: {bbox}")
    return None, bbox


class PerspectiveFaceAnimator:
    """
    Realistic face animation using perspective transforms for head rotation
    and localized deformations for facial expressions.
    """

    def __init__(self, face_image, face_app=None):
        self.original = face_image.copy()
        self.img_h, self.img_w = self.original.shape[:2]

        landmarks, bbox = detect_landmarks(self.original, face_app)
        self.landmarks = landmarks
        self.bbox = bbox

        x1, y1, x2, y2 = bbox
        self.face_cx = (x1 + x2) / 2
        self.face_cy = (y1 + y2) / 2
        self.face_w = x2 - x1
        self.face_h = y2 - y1

        if landmarks is not None and len(landmarks) >= 106:
            self._extract_features_from_landmarks()
        else:
            self._estimate_features_from_bbox()

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.blink = 0.0
        self.smile = 0.0
        self.mouth_open = 0.0
        self.zoom = 1.0

        self.micro_t = 0.0
        self.auto_blink_timer = time.time() + np.random.uniform(2, 5)
        self.blink_duration = 0.18
        self.blink_start = 0
        self.is_blinking = False
        self.breath_t = 0.0

        self.bg_color = self._detect_bg_color()

        self._prepare_source()

        if landmarks is not None:
            print(f"  Face detected: {len(landmarks)} landmarks")
        else:
            print(f"  Face detected: approximate (Haar cascade)")

    def _extract_features_from_landmarks(self):
        lm = self.landmarks
        self.left_eye_center = lm[74:78].mean(axis=0)
        self.right_eye_center = lm[68:72].mean(axis=0)
        self.left_eye_rect = self._get_eye_rect(lm[52:71])
        self.right_eye_rect = self._get_eye_rect(lm[35:51])
        mouth_pts = lm[84:104]
        mx, my = mouth_pts.min(axis=0)
        mxx, mxy = mouth_pts.max(axis=0)
        pad = 5
        self.mouth_rect = (int(mx - pad), int(my - pad),
                          int(mxx - mx + 2 * pad), int(mxy - my + 2 * pad))

    def _get_eye_rect(self, pts):
        x, y = pts.min(axis=0)
        xx, xy = pts.max(axis=0)
        pad = 8
        return (int(x - pad), int(y - pad), int(xx - x + 2 * pad), int(xy - y + 2 * pad))

    def _estimate_features_from_bbox(self):
        cx, cy = self.face_cx, self.face_cy
        fw, fh = self.face_w, self.face_h

        eye_y = cy - fh * 0.12
        self.left_eye_center = np.array([cx - fw * 0.17, eye_y])
        self.right_eye_center = np.array([cx + fw * 0.17, eye_y])

        ew, eh = int(fw * 0.22), int(fh * 0.08)
        self.left_eye_rect = (int(self.left_eye_center[0] - ew // 2),
                             int(self.left_eye_center[1] - eh // 2), ew, eh)
        self.right_eye_rect = (int(self.right_eye_center[0] - ew // 2),
                              int(self.right_eye_center[1] - eh // 2), ew, eh)

        mw, mh = int(fw * 0.35), int(fh * 0.12)
        self.mouth_rect = (int(cx - mw // 2), int(cy + fh * 0.2), mw, mh)

    def _detect_bg_color(self):
        img = self.original
        samples = []
        for y, x in [(2, 2), (2, -3), (-3, 2), (-3, -3),
                      (self.img_h // 2, 2), (self.img_h // 2, -3)]:
            samples.append(img[y, x])
        return np.median(samples, axis=0).astype(np.uint8)

    def _prepare_source(self):
        """Prepare source image with padding for rotation"""
        pad = int(max(self.img_w, self.img_h) * 0.15)
        self.pad = pad
        bg = self.bg_color.reshape(1, 1, 3)
        padded = np.full((self.img_h + 2 * pad, self.img_w + 2 * pad, 3),
                        bg, dtype=np.uint8)
        padded[pad:pad + self.img_h, pad:pad + self.img_w] = self.original

        edge_w = 20
        for i in range(edge_w):
            alpha = i / edge_w
            top = pad + i
            bot = pad + self.img_h - 1 - i
            left = pad + i
            right = pad + self.img_w - 1 - i
            padded[top, pad:pad + self.img_w] = (
                padded[top, pad:pad + self.img_w].astype(float) * alpha +
                bg.astype(float) * (1 - alpha)
            ).astype(np.uint8)
            padded[bot, pad:pad + self.img_w] = (
                padded[bot, pad:pad + self.img_w].astype(float) * alpha +
                bg.astype(float) * (1 - alpha)
            ).astype(np.uint8)
            padded[pad:pad + self.img_h, left] = (
                padded[pad:pad + self.img_h, left].astype(float) * alpha +
                bg.astype(float) * (1 - alpha)
            ).astype(np.uint8)
            padded[pad:pad + self.img_h, right] = (
                padded[pad:pad + self.img_h, right].astype(float) * alpha +
                bg.astype(float) * (1 - alpha)
            ).astype(np.uint8)

        self.padded_source = padded
        self.padded_h, self.padded_w = padded.shape[:2]
        self.padded_cx = self.face_cx + pad
        self.padded_cy = self.face_cy + pad

    def update(self, dt):
        self.micro_t += dt
        self.breath_t += dt

        now = time.time()
        if not self.is_blinking and now >= self.auto_blink_timer:
            self.is_blinking = True
            self.blink_start = now
            self.auto_blink_timer = now + np.random.uniform(2.0, 5.5)

        if self.is_blinking:
            elapsed = now - self.blink_start
            half = self.blink_duration / 2
            if elapsed < half:
                self.blink = min(1.0, elapsed / half)
            elif elapsed < self.blink_duration:
                self.blink = max(0.0, 1.0 - (elapsed - half) / half)
            else:
                self.blink = 0.0
                self.is_blinking = False

    def _apply_expressions(self, img):
        """Apply blink/smile/mouth on the source image before rotation"""
        if self.blink > 0.1:
            self._apply_blink(img, self.blink)
        if self.smile > 0.1:
            self._apply_smile(img, self.smile)
        if self.mouth_open > 0.1:
            self._apply_mouth_open(img, self.mouth_open)
        return img

    def _apply_blink(self, img, amount):
        """Close eyes by stretching skin above eye downward"""
        for eye_rect in [self.left_eye_rect, self.right_eye_rect]:
            x, y, w, h = eye_rect
            x = max(0, x)
            y = max(0, y)
            if x + w > img.shape[1] or y + h > img.shape[0]:
                continue

            lid_h = max(3, h // 3)
            lid_y = max(0, y - lid_h)
            lid = img[lid_y:y, x:x + w].copy()
            if lid.size == 0:
                continue

            cover_h = int(h * amount * 0.8)
            if cover_h < 1:
                continue

            lid_stretched = cv2.resize(lid, (w, cover_h), interpolation=cv2.INTER_LINEAR)

            end_y = min(y + cover_h, img.shape[0])
            actual_h = end_y - y
            if actual_h > 0 and actual_h <= lid_stretched.shape[0]:
                alpha_strip = np.linspace(0.9, 0.6, actual_h).reshape(-1, 1, 1)
                region = img[y:end_y, x:x + w].astype(float)
                overlay = lid_stretched[:actual_h].astype(float)
                img[y:end_y, x:x + w] = (overlay * alpha_strip + region * (1 - alpha_strip)).astype(np.uint8)

            blur_y = max(0, y - 2)
            blur_end = min(img.shape[0], end_y + 2)
            if blur_end > blur_y:
                roi = img[blur_y:blur_end, max(0, x - 1):min(img.shape[1], x + w + 1)]
                if roi.size > 0:
                    cv2.GaussianBlur(roi, (3, 3), 0.8, dst=roi)

    def _apply_smile(self, img, amount):
        """Warp mouth corners upward for smile"""
        mx, my, mw, mh = self.mouth_rect
        if mx < 0 or my < 0 or mx + mw > img.shape[1] or my + mh > img.shape[0]:
            return

        region = img[my:my + mh, mx:mx + mw].copy()
        if region.size == 0:
            return

        rh, rw = region.shape[:2]
        map_x = np.zeros((rh, rw), dtype=np.float32)
        map_y = np.zeros((rh, rw), dtype=np.float32)

        for py in range(rh):
            for px in range(rw):
                nx = (px / rw - 0.5) * 2
                ny = (py / rh - 0.5) * 2

                dx = nx * amount * 0.05
                dy = -abs(nx) * amount * 0.08 * (1 - ny * 0.3)

                map_x[py, px] = px - dx * rw / 2
                map_y[py, px] = py - dy * rh / 2

        warped = cv2.remap(region, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2 - 2, rh // 2 - 2), 0, 0, 360, 1.0, -1)
        cv2.GaussianBlur(mask, (7, 7), 3, dst=mask)
        mask3 = np.stack([mask] * 3, axis=-1)

        blended = (warped.astype(float) * mask3 +
                  region.astype(float) * (1 - mask3)).astype(np.uint8)
        img[my:my + mh, mx:mx + mw] = blended

    def _apply_mouth_open(self, img, amount):
        """Open mouth by stretching lower region and darkening center"""
        mx, my, mw, mh = self.mouth_rect
        if mx < 0 or my < 0 or mx + mw > img.shape[1] or my + mh > img.shape[0]:
            return

        cx_local = mw // 2
        cy_local = mh // 2

        open_h = int(mh * amount * 0.4)
        open_w = int(mw * 0.4)

        dark_y = my + cy_local
        dark_x = mx + cx_local

        cv2.ellipse(img, (dark_x, dark_y + int(open_h * 0.3)),
                   (open_w // 2, max(1, open_h // 2)),
                   0, 0, 360, (20, 15, 15), -1)
        cv2.GaussianBlur(
            img[max(0, dark_y - open_h):min(img.shape[0], dark_y + open_h),
                max(0, dark_x - open_w):min(img.shape[1], dark_x + open_w)],
            (5, 5), 2,
            dst=img[max(0, dark_y - open_h):min(img.shape[0], dark_y + open_h),
                    max(0, dark_x - open_w):min(img.shape[1], dark_x + open_w)]
        )

    def _get_head_pose_transform(self, yaw, pitch, roll):
        """Compute perspective homography for 3D head rotation"""
        cx, cy = self.padded_cx, self.padded_cy
        f = self.img_w * 1.2

        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)

        Ry = np.array([
            [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
            [0, 1, 0],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ])

        Rz = np.array([
            [math.cos(roll_rad), -math.sin(roll_rad), 0],
            [math.sin(roll_rad), math.cos(roll_rad), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx

        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        K_inv = np.linalg.inv(K)
        H = K @ R @ K_inv

        return H

    def render(self):
        """Render one frame"""
        micro_yaw = math.sin(self.micro_t * 0.3) * 0.4 + math.sin(self.micro_t * 0.7) * 0.2
        micro_pitch = math.sin(self.micro_t * 0.4) * 0.25 + math.cos(self.micro_t * 0.55) * 0.15
        eff_yaw = self.yaw + micro_yaw
        eff_pitch = self.pitch + micro_pitch

        breath_y = math.sin(self.breath_t * 0.8) * 0.3

        src = self.original.copy()
        src = self._apply_expressions(src)

        padded = self.padded_source.copy()
        p = self.pad
        padded[p:p + self.img_h, p:p + self.img_w] = src

        H = self._get_head_pose_transform(eff_yaw, eff_pitch + breath_y, self.roll)
        warped = cv2.warpPerspective(
            padded, H, (self.padded_w, self.padded_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        if abs(eff_yaw) > 2:
            shadow = np.ones_like(warped, dtype=np.float32)
            shadow_strength = min(0.25, abs(eff_yaw) / 60)
            cx = self.padded_w // 2
            for x in range(self.padded_w):
                if eff_yaw > 0 and x < cx:
                    factor = 1.0 - shadow_strength * (1.0 - x / cx)
                elif eff_yaw < 0 and x > cx:
                    factor = 1.0 - shadow_strength * ((x - cx) / (self.padded_w - cx))
                else:
                    factor = 1.0
                shadow[:, x, :] = factor
            warped = (warped.astype(float) * shadow).astype(np.uint8)

        scale = min(CAM_WIDTH / self.img_w, CAM_HEIGHT / self.img_h) * 0.82 * self.zoom
        crop_w = int(self.img_w * 1.3)
        crop_h = int(self.img_h * 1.3)

        cx_int = int(self.padded_cx)
        cy_int = int(self.padded_cy)
        x1 = max(0, cx_int - crop_w // 2)
        y1 = max(0, cy_int - crop_h // 2)
        x2 = min(self.padded_w, x1 + crop_w)
        y2 = min(self.padded_h, y1 + crop_h)

        cropped = warped[y1:y2, x1:x2]

        frame = np.full((CAM_HEIGHT, CAM_WIDTH, 3), self.bg_color.reshape(1, 1, 3), dtype=np.uint8)

        if cropped.size > 0:
            scaled = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            sh, sw = scaled.shape[:2]

            ox = (CAM_WIDTH - sw) // 2
            oy = (CAM_HEIGHT - sh) // 2

            src_x1 = max(0, -ox)
            src_y1 = max(0, -oy)
            dst_x1 = max(0, ox)
            dst_y1 = max(0, oy)
            copy_w = min(sw - src_x1, CAM_WIDTH - dst_x1)
            copy_h = min(sh - src_y1, CAM_HEIGHT - dst_y1)

            if copy_w > 0 and copy_h > 0:
                frame[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = \
                    scaled[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]

        frame = cv2.bilateralFilter(frame, 5, 40, 40)

        return frame


class FaceCam3DApp:
    def __init__(self, face_image=None, use_virtual_cam=True):
        print("Initializing AI Face Cam 3D...")

        self.face_app = get_face_analysis()
        self.running = True
        self.last_time = time.time()
        self.virtual_cam = None
        self.use_virtual_cam = use_virtual_cam

        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.target_roll = 0.0
        self.target_smile = 0.0
        self.smooth = 4.0

        if face_image is None:
            face_image = generate_ai_face()
            if face_image is None:
                print("Could not generate face. Please select an image.")
                face_image = self._load_from_dialog()

        if face_image is None:
            print("No face image available!")
            sys.exit(1)

        self._init_animator(face_image)
        self._init_virtual_cam()

    def _init_animator(self, face_image):
        print("Setting up face animation...")
        self.animator = PerspectiveFaceAnimator(face_image, self.face_app)

    def _init_virtual_cam(self):
        if not self.use_virtual_cam:
            return
        try:
            import pyvirtualcam
            self.virtual_cam = pyvirtualcam.Camera(
                width=CAM_WIDTH, height=CAM_HEIGHT, fps=FPS,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            print(f"Virtual camera: {self.virtual_cam.device}")
        except Exception as e:
            print(f"Virtual camera not available: {e}")
            print("Preview mode. Install OBS Studio for webcam output.")

    def _load_from_dialog(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(
                title="AI Face Cam - Select Face Photo",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
            )
            root.destroy()
            if path:
                img = cv2.imread(path)
                return img
        except Exception as e:
            print(f"File dialog error: {e}")
        return None

    def _generate_new_face(self):
        """Generate a new AI face and reinitialize"""
        face = generate_ai_face()
        if face is not None:
            self._init_animator(face)
            print("New face loaded!")
        else:
            print("Failed to generate face. Check internet connection.")

    def _load_new_image(self):
        """Load a new image from file"""
        img = self._load_from_dialog()
        if img is not None:
            self._init_animator(img)
            print("Image loaded!")
        else:
            print("No image selected.")

    def run(self):
        print("\n" + "=" * 42)
        print("  AI Face Cam 3D - RUNNING")
        print("=" * 42)
        print("  Movement:")
        print("    A/D     Turn head left/right")
        print("    W/S     Look up/down")
        print("    Q/E     Tilt head")
        print("  Expressions:")
        print("    B       Blink")
        print("    N       Smile")
        print("    M       Open/close mouth")
        print("  Other:")
        print("    G       Generate new AI face")
        print("    L       Load your own photo")
        print("    +/-     Zoom in/out")
        print("    R       Reset position")
        print("    ESC     Quit")
        if self.virtual_cam:
            print(f"\n  WEBCAM ACTIVE: {self.virtual_cam.device}")
        else:
            print("\n  Preview mode (install OBS for webcam)")
        print("=" * 42)

        cv2.namedWindow('AI Face Cam 3D', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AI Face Cam 3D', CAM_WIDTH, CAM_HEIGHT)

        frame_times = []

        while self.running:
            t0 = time.time()
            dt = t0 - self.last_time
            self.last_time = t0

            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)

            try:
                if cv2.getWindowProperty('AI Face Cam 3D', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            s = min(1.0, self.smooth * dt)
            self.animator.yaw += (self.target_yaw - self.animator.yaw) * s
            self.animator.pitch += (self.target_pitch - self.animator.pitch) * s
            self.animator.roll += (self.target_roll - self.animator.roll) * s
            self.animator.smile += (self.target_smile - self.animator.smile) * s

            self.animator.update(dt)
            frame = self.animator.render()

            frame_times.append(time.time() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            info = f"FPS:{avg_fps:.0f} Yaw:{self.animator.yaw:.0f} Pitch:{self.animator.pitch:.0f}"
            cv2.putText(frame, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            cv2.putText(frame, "G=New Face  L=Load Photo", (10, CAM_HEIGHT - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            if self.virtual_cam:
                cv2.putText(frame, "LIVE", (CAM_WIDTH - 50, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('AI Face Cam 3D', frame)

            if self.virtual_cam:
                self.virtual_cam.send(frame)

            elapsed = time.time() - t0
            wait = (1.0 / FPS) - elapsed
            if wait > 0:
                time.sleep(wait)

        cv2.destroyAllWindows()
        if self.virtual_cam:
            self.virtual_cam.close()

    def _handle_key(self, key):
        spd = 2.0
        if key == 27:
            self.running = False
        elif key in (ord('a'), ord('A')):
            self.target_yaw = max(-30, self.target_yaw - spd)
        elif key in (ord('d'), ord('D')):
            self.target_yaw = min(30, self.target_yaw + spd)
        elif key in (ord('w'), ord('W')):
            self.target_pitch = max(-20, self.target_pitch - spd)
        elif key in (ord('s'), ord('S')):
            self.target_pitch = min(20, self.target_pitch + spd)
        elif key in (ord('q'), ord('Q')):
            self.target_roll = max(-15, self.target_roll - spd)
        elif key in (ord('e'), ord('E')):
            self.target_roll = min(15, self.target_roll + spd)
        elif key in (ord('b'), ord('B')):
            self.animator.is_blinking = True
            self.animator.blink_start = time.time()
        elif key in (ord('m'), ord('M')):
            self.animator.mouth_open = 0.0 if self.animator.mouth_open > 0.3 else 0.6
        elif key in (ord('n'), ord('N')):
            self.target_smile = 0.0 if self.target_smile > 0.3 else 0.8
        elif key in (ord('+'), ord('=')):
            self.animator.zoom = min(2.0, self.animator.zoom + 0.05)
        elif key in (ord('-'), ord('_')):
            self.animator.zoom = max(0.5, self.animator.zoom - 0.05)
        elif key in (ord('r'), ord('R')):
            self.target_yaw = self.target_pitch = self.target_roll = 0
            self.target_smile = 0
            self.animator.zoom = 1.0
            self.animator.mouth_open = 0.0
        elif key in (ord('g'), ord('G')):
            self._generate_new_face()
        elif key in (ord('l'), ord('L')):
            self._load_new_image()


def select_image_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="AI Face Cam - Select Face Photo",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
        )
        root.destroy()
        return path
    except Exception:
        return None


def main():
    print("=" * 42)
    print("  AI Face Cam 3D")
    print("  Realistic face animation + webcam")
    print("=" * 42)
    print()

    ensure_dirs()
    ensure_models()

    face_image = None

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('-'):
                continue
            if os.path.isfile(arg):
                face_image = cv2.imread(arg)
                if face_image is not None:
                    print(f"Loaded: {arg}")
                break

    if face_image is None:
        print("Generating AI face (press G anytime for a new one)...")
        face_image = generate_ai_face()

    if face_image is None:
        print("Could not generate face. Select an image instead.")
        path = select_image_dialog()
        if path:
            face_image = cv2.imread(path)

    if face_image is None:
        print("\nNo face available. Exiting.")
        input("Press Enter to exit...")
        sys.exit(1)

    app = FaceCam3DApp(face_image, use_virtual_cam=True)
    app.run()


if __name__ == '__main__':
    main()
