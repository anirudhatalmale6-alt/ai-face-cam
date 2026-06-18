"""
AI Face Cam - Virtual webcam with animated AI face for KYC verification
Loads a face image and animates it with keyboard controls
Outputs through virtual camera (OBS Virtual Camera)

Usage:
  python face_cam.py -i face_photo.png
  python face_cam.py -i face_photo.png --no-virtual-cam
"""

import cv2
import numpy as np
import math
import time
import sys
import os
import argparse

CAM_WIDTH = 640
CAM_HEIGHT = 480
FPS = 30


class FaceAnimator:
    def __init__(self, face_image_path):
        self.original = cv2.imread(face_image_path, cv2.IMREAD_UNCHANGED)
        if self.original is None:
            raise ValueError(f"Cannot load image: {face_image_path}")

        # Convert RGBA to BGR if needed
        if self.original.shape[2] == 4:
            self.original = cv2.cvtColor(self.original, cv2.COLOR_BGRA2BGR)

        self.img_h, self.img_w = self.original.shape[:2]

        # Detect face using OpenCV DNN
        self.face_rect = self._detect_face()
        if self.face_rect is None:
            print("Warning: No face detected, using center of image")
            margin = min(self.img_w, self.img_h) // 8
            self.face_rect = (margin, margin, self.img_w - margin * 2, self.img_h - margin * 2)

        fx, fy, fw, fh = self.face_rect
        self.face_cx = fx + fw // 2
        self.face_cy = fy + fh // 2

        # Estimate eye positions (proportional to face)
        self.left_eye = (self.face_cx - fw // 5, fy + int(fh * 0.38))
        self.right_eye = (self.face_cx + fw // 5, fy + int(fh * 0.38))
        self.eye_w = fw // 5
        self.eye_h = fw // 12

        # Animation state
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.blink = 0.0
        self.zoom = 1.0

        # Auto blink
        self.auto_blink_timer = time.time() + np.random.uniform(2.5, 5.0)
        self.blink_duration = 0.15
        self.blink_start = 0
        self.is_blinking = False

        # Micro movement timer
        self.micro_t = 0.0

        # Background
        self.bg_color = self._detect_bg_color()

    def _detect_face(self):
        """Detect face using OpenCV's DNN face detector"""
        prototxt = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(prototxt)
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) > 0:
            # Return largest face
            areas = [w * h for (x, y, w, h) in faces]
            idx = np.argmax(areas)
            return tuple(faces[idx])
        return None

    def _detect_bg_color(self):
        """Sample corners to detect background color"""
        corners = [
            self.original[5, 5],
            self.original[5, self.img_w - 5],
            self.original[self.img_h - 5, 5],
            self.original[self.img_h - 5, self.img_w - 5]
        ]
        return np.mean(corners, axis=0).astype(np.uint8)

    def update(self, dt):
        self.micro_t += dt

        # Auto blink
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

    def render(self):
        frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg_color

        img = self.original.copy()

        # Micro-movements for realism
        micro_yaw = math.sin(self.micro_t * 0.3) * 0.4
        micro_pitch = math.sin(self.micro_t * 0.45) * 0.3
        eff_yaw = self.yaw + micro_yaw
        eff_pitch = self.pitch + micro_pitch
        eff_roll = self.roll + math.sin(self.micro_t * 0.2) * 0.2

        # Apply perspective warp
        img = self._warp_face(img, eff_yaw, eff_pitch, eff_roll)

        # Apply blink
        if self.blink > 0.05:
            img = self._apply_blink(img)

        # Scale and center
        scale = min(CAM_WIDTH / self.img_w, CAM_HEIGHT / self.img_h) * 0.85 * self.zoom
        nw = int(self.img_w * scale)
        nh = int(self.img_h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        x_off = (CAM_WIDTH - nw) // 2
        y_off = (CAM_HEIGHT - nh) // 2

        # Paste
        sx = max(0, -x_off)
        sy = max(0, -y_off)
        dx = max(0, x_off)
        dy = max(0, y_off)
        cw = min(nw - sx, CAM_WIDTH - dx)
        ch = min(nh - sy, CAM_HEIGHT - dy)
        if cw > 0 and ch > 0:
            frame[dy:dy + ch, dx:dx + cw] = resized[sy:sy + ch, sx:sx + cw]

        return frame

    def _warp_face(self, img, yaw, pitch, roll):
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2

        # Perspective warp for yaw
        if abs(yaw) > 1:
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            sq = abs(yaw) / 50.0 * 0.12
            if yaw > 0:
                dst = np.float32([[0, h * sq], [w, 0], [w, h], [0, h * (1 - sq)]])
            else:
                dst = np.float32([[0, 0], [w, h * sq], [w, h * (1 - sq)], [0, h]])
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Perspective warp for pitch
        if abs(pitch) > 1:
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            sq = abs(pitch) / 35.0 * 0.08
            if pitch > 0:
                dst = np.float32([[w * sq, 0], [w * (1 - sq), 0], [w, h], [0, h]])
            else:
                dst = np.float32([[0, 0], [w, 0], [w * (1 - sq), h], [w * sq, h]])
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Roll rotation
        dx = math.sin(math.radians(yaw)) * w * 0.04
        dy = math.sin(math.radians(pitch)) * h * 0.04
        M_rot = cv2.getRotationMatrix2D((cx, cy), -roll, 1.0)
        M_rot[0, 2] += dx
        M_rot[1, 2] += dy
        img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return img

    def _apply_blink(self, img):
        result = img.copy()
        scale = self.face_rect[2] / self.img_w if self.face_rect else 1.0

        for eye_pos in [self.left_eye, self.right_eye]:
            ex, ey = eye_pos
            ew, eh = self.eye_w, self.eye_h

            # Sample skin color around eye
            y1 = max(0, ey - eh - 5)
            y2 = min(img.shape[0], ey + eh + 5)
            x1 = max(0, ex - ew - 3)
            x2 = min(img.shape[1], ex + ew + 3)

            region = img[y1:y2, x1:x2]
            if region.size == 0:
                continue
            skin = np.median(region, axis=(0, 1)).astype(np.uint8)

            # Draw eyelid closing
            close_amount = int(self.blink * eh * 1.5)
            if close_amount > 0:
                cv2.ellipse(result, (ex, ey), (ew, close_amount),
                           0, 0, 360, skin.tolist(), -1)
                # Add eyelid line
                darker = (max(0, int(skin[0]) - 30), max(0, int(skin[1]) - 30), max(0, int(skin[2]) - 30))
                cv2.ellipse(result, (ex, ey + close_amount // 3), (ew, 2),
                           0, 0, 360, darker, 1)

        return result


class FaceCamApp:
    def __init__(self, face_image_path, use_virtual_cam=True):
        self.animator = FaceAnimator(face_image_path)
        self.running = True
        self.last_time = time.time()
        self.virtual_cam = None

        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.target_roll = 0.0
        self.smooth = 5.0

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
                print("Showing preview window only. Install OBS Virtual Camera for webcam output.")

    def run(self):
        print("\n=== AI Face Cam ===")
        print("  A/D  - Turn head left/right")
        print("  W/S  - Tilt head up/down")
        print("  Q/E  - Roll head")
        print("  B    - Blink")
        print("  +/-  - Zoom")
        print("  R    - Reset")
        print("  ESC  - Quit")

        cv2.namedWindow('AI Face Cam', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AI Face Cam', CAM_WIDTH, CAM_HEIGHT)

        while self.running:
            now = time.time()
            dt = now - self.last_time
            self.last_time = now

            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)

            # Smooth interpolation
            s = min(1.0, self.smooth * dt)
            self.animator.yaw += (self.target_yaw - self.animator.yaw) * s
            self.animator.pitch += (self.target_pitch - self.animator.pitch) * s
            self.animator.roll += (self.target_roll - self.animator.roll) * s

            self.animator.update(dt)
            frame = self.animator.render()

            # HUD
            info = f"Yaw:{self.animator.yaw:.0f} Pitch:{self.animator.pitch:.0f} Zoom:{self.animator.zoom:.1f}"
            cv2.putText(frame, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            if self.virtual_cam:
                cv2.putText(frame, "LIVE", (CAM_WIDTH - 50, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('AI Face Cam', frame)

            if self.virtual_cam:
                self.virtual_cam.send(frame)

            elapsed = time.time() - now
            wait = (1.0 / FPS) - elapsed
            if wait > 0:
                time.sleep(wait)

        cv2.destroyAllWindows()
        if self.virtual_cam:
            self.virtual_cam.close()

    def _handle_key(self, key):
        spd = 2.5
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
        elif key in (ord('+'), ord('=')):
            self.animator.zoom = min(2.0, self.animator.zoom + 0.05)
        elif key in (ord('-'), ord('_')):
            self.animator.zoom = max(0.5, self.animator.zoom - 0.05)
        elif key in (ord('r'), ord('R')):
            self.target_yaw = self.target_pitch = self.target_roll = 0
            self.animator.zoom = 1.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Face Cam')
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to face image')
    parser.add_argument('--no-virtual-cam', action='store_true', help='Preview only, no virtual camera')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"File not found: {args.image}")
        sys.exit(1)

    app = FaceCamApp(args.image, use_virtual_cam=not args.no_virtual_cam)
    app.run()
