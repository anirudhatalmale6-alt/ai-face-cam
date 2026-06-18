"""
AI Face Cam 3D - Realistic face animation with virtual webcam
Standalone Windows exe - no Python needed

Double-click to run → select face photo → animate with keyboard
"""

import cv2
import numpy as np
import math
import time
import sys
import os
import argparse
import urllib.request

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")

CAM_WIDTH = 640
CAM_HEIGHT = 480
FPS = 30

MODEL_URLS = {
    "buffalo_l/det_10g.onnx": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/det_10g.onnx",
        "size_mb": 16
    },
    "buffalo_l/2d106det.onnx": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/2d106det.onnx",
        "size_mb": 5
    },
}


def ensure_models():
    for filename, info in MODEL_URLS.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {filename} (~{info['size_mb']}MB)...")
        try:
            urllib.request.urlretrieve(info["url"], path)
            print(f"  Done: {path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  Will use fallback face detection")


def select_image_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="AI Face Cam - Select Face Photo",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return path
    except Exception as e:
        print(f"File dialog error: {e}")
        return None


def get_face_analysis():
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            root=os.path.dirname(MODELS_DIR),
            allowed_modules=['detection', 'landmark_2d_106'],
            providers=['CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        print(f"InsightFace init failed: {e}")
        print("Using OpenCV fallback face detection")
        return None


class RealisticFaceAnimator:
    def __init__(self, face_image_path):
        self.original = cv2.imread(face_image_path)
        if self.original is None:
            raise ValueError(f"Cannot load image: {face_image_path}")

        h, w = self.original.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            self.original = cv2.resize(self.original, None, fx=scale, fy=scale)

        self.img_h, self.img_w = self.original.shape[:2]

        self.face_app = get_face_analysis()
        self.landmarks_2d = None
        self.face_bbox = None

        if self.face_app:
            faces = self.face_app.get(self.original)
            if faces:
                face = faces[0]
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    self.landmarks_2d = face.landmark_2d_106
                elif hasattr(face, 'kps') and face.kps is not None:
                    self.landmarks_2d = face.kps
                self.face_bbox = face.bbox.astype(int)
                print(f"Face detected: {len(self.landmarks_2d)} landmarks")

        if self.landmarks_2d is None:
            self._fallback_detection()

        self._setup_mesh()

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

        self.bg = self._get_bg_color()

    def _fallback_detection(self):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = faces[np.argmax([w * h for (x, y, w, h) in faces])]
            self.face_bbox = np.array([x, y, x + w, y + h])
        else:
            m = min(self.img_w, self.img_h) // 8
            self.face_bbox = np.array([m, m, self.img_w - m, self.img_h - m])

        x1, y1, x2, y2 = self.face_bbox
        fw, fh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        self.landmarks_2d = np.array([
            [x1, cy], [x1 + fw * 0.1, y1 + fh * 0.15], [x1 + fw * 0.25, y1],
            [cx, y1 - fh * 0.05], [x2 - fw * 0.25, y1], [x2 - fw * 0.1, y1 + fh * 0.15], [x2, cy],
            [x2 - fw * 0.1, y2 - fh * 0.2], [x2 - fw * 0.25, y2], [cx, y2 + fh * 0.05],
            [x1 + fw * 0.25, y2], [x1 + fw * 0.1, y2 - fh * 0.2],
            [cx - fw * 0.22, cy - fh * 0.12], [cx - fw * 0.15, cy - fh * 0.15],
            [cx - fw * 0.08, cy - fh * 0.12], [cx - fw * 0.08, cy - fh * 0.08],
            [cx - fw * 0.15, cy - fh * 0.08], [cx - fw * 0.22, cy - fh * 0.08],
            [cx + fw * 0.08, cy - fh * 0.12], [cx + fw * 0.15, cy - fh * 0.15],
            [cx + fw * 0.22, cy - fh * 0.12], [cx + fw * 0.22, cy - fh * 0.08],
            [cx + fw * 0.15, cy - fh * 0.08], [cx + fw * 0.08, cy - fh * 0.08],
            [cx, cy - fh * 0.05], [cx - fw * 0.05, cy + fh * 0.05],
            [cx, cy + fh * 0.08], [cx + fw * 0.05, cy + fh * 0.05],
            [cx - fw * 0.12, cy + fh * 0.18], [cx - fw * 0.06, cy + fh * 0.15],
            [cx, cy + fh * 0.16], [cx + fw * 0.06, cy + fh * 0.15],
            [cx + fw * 0.12, cy + fh * 0.18], [cx + fw * 0.06, cy + fh * 0.22],
            [cx, cy + fh * 0.23], [cx - fw * 0.06, cy + fh * 0.22],
        ], dtype=np.float32)

        print(f"Fallback: {len(self.landmarks_2d)} approximate landmarks")

    def _setup_mesh(self):
        from scipy.spatial import Delaunay

        pts = self.landmarks_2d[:, :2].copy()

        boundary = np.array([
            [0, 0], [self.img_w // 4, 0], [self.img_w // 2, 0],
            [3 * self.img_w // 4, 0], [self.img_w - 1, 0],
            [self.img_w - 1, self.img_h // 4], [self.img_w - 1, self.img_h // 2],
            [self.img_w - 1, 3 * self.img_h // 4], [self.img_w - 1, self.img_h - 1],
            [3 * self.img_w // 4, self.img_h - 1], [self.img_w // 2, self.img_h - 1],
            [self.img_w // 4, self.img_h - 1], [0, self.img_h - 1],
            [0, 3 * self.img_h // 4], [0, self.img_h // 2], [0, self.img_h // 4]
        ], dtype=np.float32)

        self.all_pts = np.vstack([pts, boundary])
        self.n_face_pts = len(pts)

        self.tri = Delaunay(self.all_pts)
        self.simplices = self.tri.simplices

    def _get_bg_color(self):
        corners = [self.original[2, 2], self.original[2, -3],
                   self.original[-3, 2], self.original[-3, -3]]
        return np.mean(corners, axis=0).astype(np.uint8)

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

    def _get_deformed_points(self):
        pts = self.all_pts.copy()
        face_pts = pts[:self.n_face_pts]

        x1, y1, x2, y2 = self.face_bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        fw = x2 - x1
        fh = y2 - y1

        micro_yaw = math.sin(self.micro_t * 0.3) * 0.5 + math.sin(self.micro_t * 0.7) * 0.3
        micro_pitch = math.sin(self.micro_t * 0.4) * 0.3 + math.cos(self.micro_t * 0.55) * 0.2
        eff_yaw = self.yaw + micro_yaw
        eff_pitch = self.pitch + micro_pitch

        breath = math.sin(self.breath_t * 0.8) * 1.5

        for i in range(self.n_face_pts):
            px, py = face_pts[i]

            dx_norm = (px - cx) / (fw / 2) if fw > 0 else 0
            dy_norm = (py - cy) / (fh / 2) if fh > 0 else 0

            yaw_rad = math.radians(eff_yaw)
            depth = 1.0 + dx_norm * math.sin(yaw_rad) * 0.15
            px_new = cx + (px - cx) * depth + math.sin(yaw_rad) * fw * 0.04

            pitch_rad = math.radians(eff_pitch)
            depth_v = 1.0 + dy_norm * math.sin(pitch_rad) * 0.12
            py_new = cy + (py - cy) * depth_v + math.sin(pitch_rad) * fh * 0.04

            if abs(self.roll) > 0.5:
                roll_rad = math.radians(self.roll)
                rx = (px_new - cx) * math.cos(roll_rad) - (py_new - cy) * math.sin(roll_rad)
                ry = (px_new - cx) * math.sin(roll_rad) + (py_new - cy) * math.cos(roll_rad)
                px_new = cx + rx
                py_new = cy + ry

            py_new += breath

            if self.blink > 0.05:
                eye_region = (abs(dy_norm) < 0.3 and abs(dy_norm + 0.1) < 0.2)
                if eye_region and dy_norm < 0:
                    py_new += self.blink * fh * 0.03

            if self.smile > 0.05:
                if dy_norm > 0.15 and abs(dx_norm) > 0.05:
                    px_new += dx_norm * self.smile * fw * 0.03
                    py_new -= self.smile * fh * 0.02

            if self.mouth_open > 0.05:
                if dy_norm > 0.2:
                    py_new += self.mouth_open * fh * 0.04

            pts[i] = [px_new, py_new]

        return pts

    def _warp_triangle(self, src, dst_img, src_tri, dst_tri):
        sr = cv2.boundingRect(np.float32([src_tri]))
        dr = cv2.boundingRect(np.float32([dst_tri]))

        src_tri_local = [(p[0] - sr[0], p[1] - sr[1]) for p in src_tri]
        dst_tri_local = [(p[0] - dr[0], p[1] - dr[1]) for p in dst_tri]

        sx, sy, sw, sh = sr
        if sx < 0 or sy < 0 or sx + sw > src.shape[1] or sy + sh > src.shape[0]:
            return
        src_crop = src[sy:sy + sh, sx:sx + sw]
        if src_crop.size == 0:
            return

        M = cv2.getAffineTransform(
            np.float32(src_tri_local),
            np.float32(dst_tri_local)
        )

        dx, dy, dw, dh = dr
        if dw <= 0 or dh <= 0:
            return

        warped = cv2.warpAffine(src_crop, M, (dw, dh), borderMode=cv2.BORDER_REFLECT)

        mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_local), 255)

        if dy < 0 or dx < 0 or dy + dh > dst_img.shape[0] or dx + dw > dst_img.shape[1]:
            return

        dst_region = dst_img[dy:dy + dh, dx:dx + dw]
        mask3 = cv2.merge([mask, mask, mask])
        dst_region[:] = np.where(mask3 > 0, warped, dst_region)

    def render(self):
        frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg

        dst_pts = self._get_deformed_points()
        src_pts = self.all_pts

        scale = min(CAM_WIDTH / self.img_w, CAM_HEIGHT / self.img_h) * 0.85 * self.zoom
        offset_x = (CAM_WIDTH - self.img_w * scale) / 2
        offset_y = (CAM_HEIGHT - self.img_h * scale) / 2

        dst_scaled = dst_pts.copy()
        dst_scaled[:, 0] = dst_pts[:, 0] * scale + offset_x
        dst_scaled[:, 1] = dst_pts[:, 1] * scale + offset_y

        for simplex in self.simplices:
            src_tri = src_pts[simplex].tolist()
            dst_tri = dst_scaled[simplex].tolist()

            try:
                self._warp_triangle(self.original, frame,
                                   [(int(p[0]), int(p[1])) for p in src_tri],
                                   [(int(p[0]), int(p[1])) for p in dst_tri])
            except Exception:
                pass

        frame = cv2.GaussianBlur(frame, (3, 3), 0.5)

        return frame


class FaceCam3DApp:
    def __init__(self, face_image_path, use_virtual_cam=True):
        print("Loading face...")
        self.animator = RealisticFaceAnimator(face_image_path)
        self.running = True
        self.last_time = time.time()
        self.virtual_cam = None

        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.target_roll = 0.0
        self.target_smile = 0.0
        self.smooth = 4.0

        if use_virtual_cam:
            try:
                import pyvirtualcam
                self.virtual_cam = pyvirtualcam.Camera(
                    width=CAM_WIDTH, height=CAM_HEIGHT, fps=FPS,
                    fmt=pyvirtualcam.PixelFormat.BGR
                )
                print(f"Virtual camera: {self.virtual_cam.device}")
            except Exception as e:
                print(f"Virtual camera not available: {e}")
                print("Preview only mode. Install OBS Studio for webcam output.")

        print("Ready!")

    def run(self):
        print("\n=== AI Face Cam 3D ===")
        print("Controls:")
        print("  A/D     Turn head left/right")
        print("  W/S     Look up/down")
        print("  Q/E     Tilt head")
        print("  B       Blink")
        print("  M       Open/close mouth")
        print("  N       Smile")
        print("  +/-     Zoom in/out")
        print("  R       Reset position")
        print("  ESC     Quit")
        if self.virtual_cam:
            print(f"\n  Virtual cam ACTIVE: {self.virtual_cam.device}")
        else:
            print("\n  No virtual cam - preview only")
            print("  Install OBS Studio for webcam output")

        cv2.namedWindow('AI Face Cam 3D', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AI Face Cam 3D', CAM_WIDTH, CAM_HEIGHT)

        frame_times = []

        while self.running:
            t0 = time.time()
            dt = t0 - self.last_time
            self.last_time = t0

            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)

            if cv2.getWindowProperty('AI Face Cam 3D', cv2.WND_PROP_VISIBLE) < 1:
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

            info = f"Yaw:{self.animator.yaw:.0f} Pitch:{self.animator.pitch:.0f} FPS:{avg_fps:.0f}"
            cv2.putText(frame, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

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
            self.target_yaw = max(-35, self.target_yaw - spd)
        elif key in (ord('d'), ord('D')):
            self.target_yaw = min(35, self.target_yaw + spd)
        elif key in (ord('w'), ord('W')):
            self.target_pitch = max(-25, self.target_pitch - spd)
        elif key in (ord('s'), ord('S')):
            self.target_pitch = min(25, self.target_pitch + spd)
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


def main():
    print("=" * 45)
    print("  AI Face Cam 3D")
    print("  Realistic face animation for webcam")
    print("=" * 45)
    print()

    ensure_models()

    image_path = None

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('-'):
                continue
            if os.path.isfile(arg):
                image_path = arg
                break

    if not image_path:
        print("Select a face photo...")
        image_path = select_image_dialog()

    if not image_path:
        print("\nNo image selected. Exiting.")
        print("Usage: drag a face photo onto this exe, or run and select one.")
        input("Press Enter to exit...")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"\nFile not found: {image_path}")
        input("Press Enter to exit...")
        sys.exit(1)

    print(f"Image: {image_path}")
    app = FaceCam3DApp(image_path, use_virtual_cam=True)
    app.run()


if __name__ == '__main__':
    main()
