"""
AI Face Cam - LivePortrait ONNX v2
Real-time face animation using LivePortrait ONNX models.
Smooth interpolation, natural idle behavior, expression blending.
No PyTorch needed - uses onnxruntime + numpy only.
"""
import os
import sys
import time
import math
import urllib.request
import numpy as np
import cv2
import glob
import argparse

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)

try:
    import pyvirtualcam
    HAS_VCAM = True
except ImportError:
    HAS_VCAM = False

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(os.path.expanduser("~"), ".aifacecam", "liveportrait_onnx")
FACES_DIR = os.path.join(os.path.expanduser("~"), ".aifacecam", "faces")

HF_BASE = "https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx"
MODEL_FILES = {
    "appearance_feature_extractor.onnx": f"{HF_BASE}/appearance_feature_extractor.onnx",
    "motion_extractor.onnx": f"{HF_BASE}/motion_extractor.onnx",
    "warping_spade-fix.onnx": f"{HF_BASE}/warping_spade-fix.onnx",
    "stitching.onnx": f"{HF_BASE}/stitching.onnx",
    "stitching_eye.onnx": f"{HF_BASE}/stitching_eye.onnx",
    "stitching_lip.onnx": f"{HF_BASE}/stitching_lip.onnx",
    "retinaface_det_static.onnx": f"{HF_BASE}/retinaface_det_static.onnx",
    "face_2dpose_106_static.onnx": f"{HF_BASE}/face_2dpose_106_static.onnx",
}


def download_model(name, url, dest_dir):
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest):
        return dest
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {name}...")
    tmp = dest + ".tmp"

    def progress(count, block_size, total_size):
        if total_size > 0:
            mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            pct = min(100, int(count * block_size * 100 / total_size))
            sys.stdout.write(f"\r  {pct}% ({mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        os.rename(tmp, dest)
        print(f"\n  Done: {name}")
        return dest
    except Exception as e:
        print(f"\n  Failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return None


def download_all_models():
    print("Checking AI models...")
    for name, url in MODEL_FILES.items():
        path = download_model(name, url, MODELS_DIR)
        if path is None:
            print(f"ERROR: Could not download {name}")
            print("Check your internet connection and try again.")
            return False
    print("All models ready!\n")
    return True


def generate_ai_face():
    os.makedirs(FACES_DIR, exist_ok=True)
    print("Generating AI face from thispersondoesnotexist.com...")
    try:
        req = urllib.request.Request(
            "https://thispersondoesnotexist.com",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        data = urllib.request.urlopen(req).read()
        face_path = os.path.join(FACES_DIR, f"face_{int(time.time())}.jpg")
        with open(face_path, "wb") as f:
            f.write(data)
        print(f"  Saved: {face_path}")
        return face_path
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def headpose_pred_to_degree(pred):
    if pred.ndim > 1 and pred.shape[1] == 66:
        idx_array = np.arange(0, 66, dtype=np.float32)
        pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        pred = pred / np.sum(pred, axis=1, keepdims=True)
        degree = np.sum(pred * idx_array, axis=1) * 3 - 97.5
        return degree
    return pred.flatten()


def get_rotation_matrix(pitch_, yaw_, roll_):
    PI = np.pi
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    if np.isscalar(pitch):
        pitch = np.array([[pitch]])
        yaw = np.array([[yaw]])
        roll = np.array([[roll]])
    elif pitch.ndim == 1:
        pitch = pitch[:, None]
        yaw = yaw[:, None]
        roll = roll[:, None]

    bs = pitch.shape[0]
    ones = np.ones([bs, 1])
    zeros = np.zeros([bs, 1])

    rot_x = np.concatenate([
        ones, zeros, zeros,
        zeros, np.cos(pitch), -np.sin(pitch),
        zeros, np.sin(pitch), np.cos(pitch)
    ], axis=1).reshape([bs, 3, 3])

    rot_y = np.concatenate([
        np.cos(yaw), zeros, np.sin(yaw),
        zeros, ones, zeros,
        -np.sin(yaw), zeros, np.cos(yaw)
    ], axis=1).reshape([bs, 3, 3])

    rot_z = np.concatenate([
        np.cos(roll), -np.sin(roll), zeros,
        np.sin(roll), np.cos(roll), zeros,
        zeros, zeros, ones
    ], axis=1).reshape([bs, 3, 3])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))
    return np.transpose(rot, (0, 2, 1))


def calc_eye_close_ratio(lmk):
    def dist(a, b):
        return np.linalg.norm(lmk[:, a] - lmk[:, b], axis=1, keepdims=True)
    left = dist(6, 18) / (dist(0, 12) + 1e-6)
    right = dist(30, 42) / (dist(24, 36) + 1e-6)
    return np.concatenate([left, right], axis=1)


def calc_lip_close_ratio(lmk):
    def dist(a, b):
        return np.linalg.norm(lmk[:, a] - lmk[:, b], axis=1, keepdims=True)
    return dist(90, 102) / (dist(48, 66) + 1e-6)


class SmoothValue:
    """Smooth interpolation for a floating point value"""
    def __init__(self, value=0.0, speed=5.0):
        self.current = value
        self.target = value
        self.speed = speed

    def set_target(self, target):
        self.target = target

    def update(self, dt):
        diff = self.target - self.current
        self.current += diff * min(1.0, self.speed * dt)
        return self.current

    def snap(self, value):
        self.current = value
        self.target = value


class NaturalBehavior:
    """Generates natural idle behavior - breathing, micro-movements, blinking"""
    def __init__(self):
        self.t = 0.0
        self.blink_value = 0.0
        self.next_blink = np.random.uniform(2.5, 6.0)
        self.blink_phase = 0
        self.blink_speed = 0.0
        self.double_blink = False
        self.double_blink_pause = 0.0

        self.breath_rate = np.random.uniform(0.18, 0.25)
        self.breath_phase = 0.0

        self.micro_yaw = 0.0
        self.micro_pitch = 0.0
        self.micro_roll = 0.0
        self.micro_targets = [0.0, 0.0, 0.0]
        self.micro_timer = 0.0
        self.micro_interval = np.random.uniform(1.5, 4.0)

        self.smile_value = 0.0
        self.smile_target = 0.0
        self.next_smile = np.random.uniform(8.0, 20.0)

    def update(self, dt):
        self.t += dt

        # Breathing - subtle pitch oscillation
        self.breath_phase += dt * self.breath_rate * 2 * math.pi
        breath_pitch = math.sin(self.breath_phase) * 0.6
        breath_scale = 1.0 + math.sin(self.breath_phase) * 0.003

        # Blinking
        self._update_blink(dt)

        # Micro-movements
        self._update_micro(dt)

        # Occasional subtle smile
        self._update_smile(dt)

        return {
            "blink": self.blink_value,
            "breath_pitch": breath_pitch,
            "breath_scale": breath_scale,
            "micro_yaw": self.micro_yaw,
            "micro_pitch": self.micro_pitch + breath_pitch,
            "micro_roll": self.micro_roll,
            "smile": self.smile_value,
        }

    def _update_blink(self, dt):
        if self.blink_phase == 0:
            self.next_blink -= dt
            if self.next_blink <= 0:
                self.blink_phase = 1
                self.blink_speed = np.random.uniform(8.0, 14.0)
                self.double_blink = np.random.random() < 0.15
                self.next_blink = np.random.uniform(2.5, 7.0)
        elif self.blink_phase == 1:  # closing
            self.blink_value += dt * self.blink_speed
            if self.blink_value >= 1.0:
                self.blink_value = 1.0
                self.blink_phase = 2
        elif self.blink_phase == 2:  # opening
            self.blink_value -= dt * self.blink_speed * 0.8
            if self.blink_value <= 0.0:
                self.blink_value = 0.0
                if self.double_blink:
                    self.double_blink = False
                    self.blink_phase = 3
                    self.double_blink_pause = 0.08
                else:
                    self.blink_phase = 0
        elif self.blink_phase == 3:  # pause before double blink
            self.double_blink_pause -= dt
            if self.double_blink_pause <= 0:
                self.blink_phase = 1
                self.blink_speed = np.random.uniform(10.0, 16.0)

    def _update_micro(self, dt):
        self.micro_timer += dt
        if self.micro_timer >= self.micro_interval:
            self.micro_timer = 0
            self.micro_interval = np.random.uniform(1.5, 4.0)
            self.micro_targets = [
                np.random.uniform(-1.2, 1.2),
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(-0.4, 0.4),
            ]

        smooth = min(1.0, 2.0 * dt)
        self.micro_yaw += (self.micro_targets[0] - self.micro_yaw) * smooth
        self.micro_pitch += (self.micro_targets[1] - self.micro_pitch) * smooth
        self.micro_roll += (self.micro_targets[2] - self.micro_roll) * smooth

    def _update_smile(self, dt):
        self.next_smile -= dt
        if self.next_smile <= 0:
            self.smile_target = np.random.uniform(0.0, 0.4) if self.smile_target < 0.1 else 0.0
            self.next_smile = np.random.uniform(5.0, 15.0)

        smooth = min(1.0, 1.5 * dt)
        self.smile_value += (self.smile_target - self.smile_value) * smooth

    def force_blink(self):
        if self.blink_phase == 0:
            self.blink_phase = 1
            self.blink_speed = np.random.uniform(8.0, 12.0)
            self.double_blink = False


class LivePortraitEngine:
    def __init__(self):
        self.sessions = {}
        self.src_info = None
        self._debug_printed = False

    def load_models(self):
        print("Loading AI models (this may take a moment)...")
        providers = ort.get_available_providers()
        use_providers = []
        if 'CUDAExecutionProvider' in providers:
            use_providers.append('CUDAExecutionProvider')
            print("  Using NVIDIA GPU acceleration!")
        if 'DmlExecutionProvider' in providers:
            use_providers.append('DmlExecutionProvider')
            print("  Using DirectML GPU acceleration!")
        use_providers.append('CPUExecutionProvider')

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        plugin_dll = None
        if getattr(sys, 'frozen', False):
            candidate = os.path.join(sys._MEIPASS, "grid_sample_3d_ort.dll")
            if os.path.exists(candidate):
                plugin_dll = candidate
        if plugin_dll is None:
            candidate = os.path.join(BASE_DIR, "grid_sample_3d_ort.dll")
            if os.path.exists(candidate):
                plugin_dll = candidate
        if plugin_dll is None:
            candidate = os.path.join(BASE_DIR, "grid_sample_3d_ort.so")
            if os.path.exists(candidate):
                plugin_dll = candidate

        if plugin_dll:
            try:
                opts.register_custom_ops_library(plugin_dll)
                print(f"  Registered GridSample3D custom op")
            except Exception as e:
                print(f"  Warning: Could not register custom op: {e}")
        else:
            print("  Warning: grid_sample_3d_ort not found!")

        for name in MODEL_FILES:
            if name.endswith((".dll", ".so")):
                continue
            path = os.path.join(MODELS_DIR, name)
            short = name.replace(".onnx", "")
            print(f"  Loading {short}...")
            self.sessions[short] = ort.InferenceSession(path, opts, providers=use_providers)

        warp_sess = self.sessions.get("warping_spade-fix")
        if warp_sess:
            print("  Warping model inputs:")
            for inp in warp_sess.get_inputs():
                print(f"    {inp.name}: {inp.shape} ({inp.type})")
        print("All models loaded!\n")

    def _run_model(self, name, *inputs):
        sess = self.sessions[name]
        feed = {}
        for i, inp in enumerate(sess.get_inputs()):
            data = inputs[i]
            if inp.type == 'tensor(float16)':
                feed[inp.name] = data.astype(np.float16)
            else:
                feed[inp.name] = data.astype(np.float32)
        return sess.run(None, feed)

    def _init_face_detector(self):
        if not hasattr(self, '_face_cascade'):
            proto = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self._face_cascade = cv2.CascadeClassifier(proto)

    def detect_face(self, img_bgr):
        self._init_face_detector()
        h, w = img_bgr.shape[:2]

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 0:
            faces = self._face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))

        if len(faces) == 0:
            return None

        areas = [fw * fh for (fx, fy, fw, fh) in faces]
        best = np.argmax(areas)
        fx, fy, fw, fh = faces[best]

        cx, cy = fx + fw // 2, fy + fh // 2
        size = int(max(fw, fh) * 1.5)
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w, cx + size // 2)
        y2 = min(h, cy + size // 2)

        face_crop = img_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_192 = cv2.resize(face_crop, (192, 192))
        face_rgb = cv2.cvtColor(face_192, cv2.COLOR_BGR2RGB)
        lmk_input = face_rgb.astype(np.float32).transpose(2, 0, 1)[None]

        lmk_out = self._run_model("face_2dpose_106_static", lmk_input)
        lmk = lmk_out[0].reshape(106, 2)

        lmk[:, 0] = (lmk[:, 0] + 1) / 2 * 192
        lmk[:, 1] = (lmk[:, 1] + 1) / 2 * 192

        lmk[:, 0] = lmk[:, 0] / 192 * (x2 - x1) + x1
        lmk[:, 1] = lmk[:, 1] / 192 * (y2 - y1) + y1

        return lmk

    def crop_face(self, img_rgb, lmk, scale=2.3, target_size=256):
        cx = np.mean(lmk[:, 0])
        cy = np.mean(lmk[:, 1])
        face_w = (np.max(lmk[:, 0]) - np.min(lmk[:, 0])) * scale
        face_h = (np.max(lmk[:, 1]) - np.min(lmk[:, 1])) * scale
        size = max(face_w, face_h)

        src_pts = np.float32([
            [cx - size / 2, cy - size / 2],
            [cx + size / 2, cy - size / 2],
            [cx - size / 2, cy + size / 2]
        ])
        dst_pts = np.float32([
            [0, 0],
            [target_size, 0],
            [0, target_size]
        ])

        M = cv2.getAffineTransform(src_pts, dst_pts)
        M_inv = cv2.getAffineTransform(dst_pts, src_pts)
        crop = cv2.warpAffine(img_rgb, M, (target_size, target_size), flags=cv2.INTER_LINEAR)

        return crop, M, M_inv

    def prepare_source(self, source_img_bgr):
        print("Processing source face...")
        img_rgb = cv2.cvtColor(source_img_bgr, cv2.COLOR_BGR2RGB)

        lmk = self.detect_face(source_img_bgr)
        if lmk is None:
            print("  No face detected in source image!")
            return False

        crop_256, M_crop, M_inv = self.crop_face(img_rgb, lmk)
        crop_input = (crop_256.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        f_s = self._run_model("appearance_feature_extractor", crop_input)[0]
        print(f"  Appearance features: {f_s.shape}")

        mot_out = self._run_model("motion_extractor", crop_input)
        pitch, yaw, roll, t, exp, scale, kp = mot_out

        pitch = headpose_pred_to_degree(pitch)
        yaw = headpose_pred_to_degree(yaw)
        roll = headpose_pred_to_degree(roll)
        kp = kp.reshape(1, -1, 3)
        exp = exp.reshape(1, -1, 3)
        num_kp = kp.shape[1]

        R_s = get_rotation_matrix(pitch, yaw, roll)
        x_s = scale[..., None] * (kp @ R_s + exp) + t[:, None, :]

        mask_crop = np.ones((256, 256, 3), dtype=np.float32)
        cv2.ellipse(mask_crop, (128, 128), (120, 140), 0, 0, 360, (1, 1, 1), -1)
        mask_crop = cv2.GaussianBlur(mask_crop, (51, 51), 20)

        h, w = source_img_bgr.shape[:2]

        lmk_batch = lmk[None]
        src_eye_ratio = calc_eye_close_ratio(lmk_batch)
        src_lip_ratio = calc_lip_close_ratio(lmk_batch)

        self.src_info = {
            "f_s": f_s,
            "x_s": x_s,
            "kp": kp,
            "exp": exp,
            "scale": scale,
            "t": t,
            "R_s": R_s,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "num_kp": num_kp,
            "crop_256": crop_256,
            "img_rgb": img_rgb,
            "M_inv": M_inv,
            "mask_crop": mask_crop,
            "orig_shape": (h, w),
            "lmk": lmk,
            "eye_ratio": src_eye_ratio,
            "lip_ratio": src_lip_ratio,
        }

        print(f"  Source ready! Keypoints: {num_kp}, Pose: pitch={pitch[0]:.1f} yaw={yaw[0]:.1f} roll={roll[0]:.1f}")
        return True

    def animate_keyboard(self, pitch_delta=0, yaw_delta=0, roll_delta=0, exp_scale=1.0, blink=0, smile=0, mouth=0):
        if self.src_info is None:
            return None

        src = self.src_info
        num_kp = src["num_kp"]

        R_delta = get_rotation_matrix(
            np.array([pitch_delta]),
            np.array([yaw_delta]),
            np.array([roll_delta])
        )
        R_new = R_delta @ src["R_s"]

        x_d_new = src["scale"][..., None] * (src["kp"] @ R_new + src["exp"]) + src["t"][:, None, :]

        feat_stitch = np.concatenate([
            src["x_s"].reshape(1, -1),
            x_d_new.reshape(1, -1)
        ], axis=1).astype(np.float32)
        delta = self._run_model("stitching", feat_stitch)[0]
        x_d_new = x_d_new + delta[..., :3 * num_kp].reshape(1, num_kp, 3)
        x_d_new[..., :2] += delta[..., 3 * num_kp:3 * num_kp + 2].reshape(1, 1, 2)

        if blink > 0:
            drv_eye = np.array([[src["eye_ratio"][0, 0] * (1 - blink)]])
            eye_input = np.concatenate([
                src["x_s"].reshape(1, -1),
                src["eye_ratio"],
                drv_eye
            ], axis=1).astype(np.float32)
            eye_delta = self._run_model("stitching_eye", eye_input)[0]
            x_d_new = x_d_new + eye_delta.reshape(1, num_kp, 3)

        if smile > 0 or mouth > 0:
            drv_lip = np.array([[src["lip_ratio"][0, 0] + max(smile, mouth) * 0.15]])
            lip_input = np.concatenate([
                src["x_s"].reshape(1, -1),
                src["lip_ratio"],
                drv_lip
            ], axis=1).astype(np.float32)
            lip_delta = self._run_model("stitching_lip", lip_input)[0]
            x_d_new = x_d_new + lip_delta.reshape(1, num_kp, 3)

        out = self._run_model(
            "warping_spade-fix",
            src["f_s"],
            x_d_new.astype(np.float32),
            src["x_s"].astype(np.float32)
        )[0]

        out_t = out[0].transpose(1, 2, 0)

        if not self._debug_printed:
            self._debug_printed = True
            print(f"  [DEBUG] warp output range: [{out.min():.4f}, {out.max():.4f}], mean={out.mean():.4f}")

        out_img = np.clip(out_t, 0, 1) * 255
        return out_img.astype(np.uint8)


def list_saved_faces():
    os.makedirs(FACES_DIR, exist_ok=True)
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    faces = []
    for ext in exts:
        faces.extend(glob.glob(os.path.join(FACES_DIR, ext)))
    faces.sort(key=os.path.getmtime, reverse=True)
    return faces


def save_face_to_gallery(img_path):
    os.makedirs(FACES_DIR, exist_ok=True)
    name = os.path.basename(img_path)
    dest = os.path.join(FACES_DIR, name)
    if not os.path.exists(dest):
        import shutil
        shutil.copy2(img_path, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description='AI Face Cam - LivePortrait ONNX v2')
    parser.add_argument('image', nargs='?', help='Path to source face image')
    parser.add_argument('--no-virtual-cam', action='store_true', help='Disable virtual camera output')
    parser.add_argument('--resolution', type=int, default=512, choices=[256, 512], help='Output resolution')
    parser.add_argument('--hide-hud', action='store_true', help='Hide the on-screen HUD info')
    args = parser.parse_args()

    RES = args.resolution

    print("=" * 50)
    print("  AI Face Cam - LivePortrait ONNX v2")
    print("  Smooth animation with natural behavior")
    print("=" * 50)
    print()

    if not download_all_models():
        input("Press Enter to exit...")
        return

    engine = LivePortraitEngine()
    engine.load_models()

    source_path = args.image

    if source_path is None:
        saved = list_saved_faces()
        print("No source face provided. Options:")
        print("  G = Generate random AI face")
        print("  C = Choose from gallery" + (f" ({len(saved)} saved)" if saved else " (generate new)"))
        print("  L = Load a photo from file")
        if saved:
            print(f"  1-9 = Quick pick from recent faces")
            for i, fp in enumerate(saved[:9]):
                print(f"    {i+1}: {os.path.basename(fp)}")
        print()
        choice = input("Choose: ").strip().upper()

        if choice == "L":
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                source_path = filedialog.askopenfilename(
                    title="Select face photo",
                    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
                )
                root.destroy()
                if source_path:
                    save_face_to_gallery(source_path)
            except:
                source_path = input("Enter path to face photo: ").strip().strip('"')
                if source_path and os.path.exists(source_path):
                    save_face_to_gallery(source_path)
        elif choice == "C":
            if saved:
                num_faces = len(saved)
            else:
                num_faces = 18
                print(f"Generating {num_faces} AI faces...")
                for i in range(num_faces):
                    sys.stdout.write(f"\r  Downloading face {i+1}/{num_faces}...")
                    sys.stdout.flush()
                    generate_ai_face()
                print()
                saved = list_saved_faces()
                num_faces = len(saved)

            if saved:
                per_page = 6
                pages = (len(saved) + per_page - 1) // per_page
                page = 0
                thumbs = []
                for p in saved:
                    t = cv2.imread(p)
                    if t is not None:
                        thumbs.append(cv2.resize(t, (170, 170)))
                    else:
                        thumbs.append(np.zeros((170, 170, 3), dtype=np.uint8))
                while True:
                    gallery = np.zeros((420, 530, 3), dtype=np.uint8)
                    start = page * per_page
                    end = min(start + per_page, len(thumbs))
                    cv2.putText(gallery, f"Page {page+1}/{pages} - Press 1-6 to pick, A/D for pages, ESC to cancel",
                               (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    for i in range(start, end):
                        idx = i - start
                        row, col = idx // 3, idx % 3
                        y0, x0 = 35 + row * 185, 5 + col * 175
                        gallery[y0:y0+170, x0:x0+170] = thumbs[i]
                        cv2.putText(gallery, str(idx+1), (x0+5, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(gallery, os.path.basename(saved[i])[:18], (x0+5, y0+165),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                    cv2.imshow("AI Face Cam", gallery)
                    k = cv2.waitKey(100) & 0xFF
                    if k >= ord('1') and k <= ord('6'):
                        sel = start + (k - ord('1'))
                        if sel < len(saved):
                            source_path = saved[sel]
                            print(f"  Selected: {os.path.basename(source_path)}")
                            break
                    elif k in (ord('d'), ord('D'), 83):
                        page = min(pages - 1, page + 1)
                    elif k in (ord('a'), ord('A'), 81):
                        page = max(0, page - 1)
                    elif k == 27:
                        break
        elif choice.isdigit() and 1 <= int(choice) <= min(9, len(saved)):
            source_path = saved[int(choice) - 1]
            print(f"  Selected: {os.path.basename(source_path)}")
        elif choice == "G":
            source_path = generate_ai_face()
        else:
            source_path = generate_ai_face()

    if not source_path or not os.path.exists(source_path):
        print("No valid source image!")
        input("Press Enter to exit...")
        return

    source_img = cv2.imread(source_path)
    if source_img is None:
        print(f"Could not read: {source_path}")
        input("Press Enter to exit...")
        return

    if not engine.prepare_source(source_img):
        input("Press Enter to exit...")
        return

    vcam = None
    if HAS_VCAM and not args.no_virtual_cam:
        try:
            vcam = pyvirtualcam.Camera(width=RES, height=RES, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Virtual camera: {vcam.device}")
        except Exception as e:
            print(f"Virtual camera not available: {e}")
            print("Install OBS Studio for virtual camera output.")

    # Pre-render poses with finer granularity
    def prerender_poses(eng):
        yaw_step = 5
        pitch_step = 5
        yaw_vals = np.arange(-30, 31, yaw_step)
        pitch_vals = np.arange(-20, 21, pitch_step)
        grid = {}
        specials = {}

        expr_combos = [
            ("blink", {"blink": 1.0}),
            ("blink_half", {"blink": 0.5}),
            ("smile", {"smile": 1.0}),
            ("smile_half", {"smile": 0.5}),
            ("mouth", {"mouth": 1.0}),
            ("mouth_half", {"mouth": 0.5}),
            ("blink+smile", {"blink": 1.0, "smile": 0.6}),
        ]

        total = len(yaw_vals) * len(pitch_vals) + len(expr_combos)
        count = 0
        print(f"\nPre-rendering {total} frames ({len(yaw_vals)}x{len(pitch_vals)} poses + {len(expr_combos)} expressions)...")
        print("This takes a few minutes. After this, animation is INSTANT.\n")

        for p in pitch_vals:
            for y in yaw_vals:
                out = eng.animate_keyboard(float(p), float(y), 0)
                if out is not None:
                    grid[(int(p), int(y))] = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                    cv2.imshow("AI Face Cam", grid[(int(p), int(y))])
                    cv2.waitKey(1)
                count += 1
                pct = count * 100 // total
                sys.stdout.write(f"\r  [{('#' * (pct//5))}{('-' * (20-pct//5))}] {pct}% ({count}/{total})")
                sys.stdout.flush()

        for label, kwargs in expr_combos:
            out = eng.animate_keyboard(0, 0, 0, **kwargs)
            if out is not None:
                specials[label] = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            count += 1
            pct = count * 100 // total
            sys.stdout.write(f"\r  [{('#' * (pct//5))}{('-' * (20-pct//5))}] {pct}% ({count}/{total})")
            sys.stdout.flush()

        # Pre-render blink at a few yaw positions for smooth blinking during head turns
        print("\n  Pre-rendering expressions at different head positions...")
        extra = 0
        for yaw_off in [-15, -5, 5, 15]:
            for blink_lbl, blink_val in [("blink", 1.0), ("blink_half", 0.5)]:
                out = eng.animate_keyboard(0, float(yaw_off), 0, blink=blink_val)
                if out is not None:
                    specials[f"{blink_lbl}_y{yaw_off}"] = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                    extra += 1

        print(f"\n  Done! {len(grid)} pose frames + {len(specials)} expressions ready.\n")
        return grid, specials, yaw_vals, pitch_vals

    grid, specials, yaw_vals, pitch_vals = prerender_poses(engine)

    print("Controls:")
    print("  A/D  = Turn head left/right")
    print("  W/S  = Look up/down")
    print("  Q/E  = Tilt head")
    print("  B    = Manual blink")
    print("  N    = Smile toggle")
    print("  M    = Open mouth toggle")
    print("  P    = Auto movement (natural)")
    print("  H    = Toggle HUD")
    print("  G    = New AI face")
    print("  L    = Load photo from file")
    print("  R    = Reset to center")
    print("  +/-  = Adjust movement speed")
    print("  ESC  = Quit")
    print()

    # Smooth state
    yaw_smooth = SmoothValue(0.0, speed=6.0)
    pitch_smooth = SmoothValue(0.0, speed=6.0)
    roll_smooth = SmoothValue(0.0, speed=4.0)

    target_yaw = 0.0
    target_pitch = 0.0
    target_roll = 0.0
    move_speed = 2.5

    behavior = NaturalBehavior()
    auto_mode = False
    auto_t = 0.0
    show_hud = not args.hide_hud

    last_time = time.time()

    # Auto movement pattern generator
    class AutoMovement:
        def __init__(self):
            self.t = 0.0
            self.freq1 = np.random.uniform(0.08, 0.15)
            self.freq2 = np.random.uniform(0.04, 0.08)
            self.freq3 = np.random.uniform(0.02, 0.05)
            self.amp_yaw = np.random.uniform(12, 20)
            self.amp_pitch = np.random.uniform(5, 10)

        def update(self, dt):
            self.t += dt
            yaw = (math.sin(self.t * self.freq1 * 2 * math.pi) * self.amp_yaw * 0.6
                   + math.sin(self.t * self.freq2 * 2 * math.pi) * self.amp_yaw * 0.3
                   + math.sin(self.t * self.freq3 * 2 * math.pi) * self.amp_yaw * 0.1)
            pitch = (math.sin(self.t * self.freq1 * 1.3 * 2 * math.pi + 1.2) * self.amp_pitch * 0.5
                     + math.sin(self.t * self.freq2 * 0.7 * 2 * math.pi + 0.8) * self.amp_pitch * 0.5)
            return yaw, pitch

    auto_mover = AutoMovement()

    def get_blended_frame(yaw_val, pitch_val, blink_val=0.0, smile_val=0.0):
        """Get frame with bilinear interpolation between pre-rendered poses"""
        yaw_min, yaw_max = float(yaw_vals[0]), float(yaw_vals[-1])
        pitch_min, pitch_max = float(pitch_vals[0]), float(pitch_vals[-1])
        yaw_clamped = max(yaw_min, min(yaw_max, yaw_val))
        pitch_clamped = max(pitch_min, min(pitch_max, pitch_val))

        yaw_step = float(yaw_vals[1] - yaw_vals[0])
        pitch_step = float(pitch_vals[1] - pitch_vals[0])

        yaw_idx = (yaw_clamped - yaw_min) / yaw_step
        pitch_idx = (pitch_clamped - pitch_min) / pitch_step

        y0 = int(math.floor(yaw_idx))
        y1 = min(y0 + 1, len(yaw_vals) - 1)
        p0 = int(math.floor(pitch_idx))
        p1 = min(p0 + 1, len(pitch_vals) - 1)

        fy = yaw_idx - y0
        fp = pitch_idx - p0

        def get_grid(pi, yi):
            p = int(pitch_vals[np.clip(pi, 0, len(pitch_vals)-1)])
            y = int(yaw_vals[np.clip(yi, 0, len(yaw_vals)-1)])
            return grid.get((p, y))

        f00 = get_grid(p0, y0)
        f01 = get_grid(p0, y1)
        f10 = get_grid(p1, y0)
        f11 = get_grid(p1, y1)

        if f00 is None:
            return grid.get((0, 0))

        # Bilinear interpolation
        if f01 is None: f01 = f00
        if f10 is None: f10 = f00
        if f11 is None: f11 = f00

        top = cv2.addWeighted(f00, 1 - fy, f01, fy, 0)
        bottom = cv2.addWeighted(f10, 1 - fy, f11, fy, 0)
        result = cv2.addWeighted(top, 1 - fp, bottom, fp, 0)

        # Blend in blink expression
        if blink_val > 0.05:
            # Find best blink frame for current yaw
            best_blink = None
            yaw_int = int(round(yaw_clamped))
            closest_yaw_key = None
            min_dist = 999

            for key in specials:
                if key.startswith("blink_y"):
                    ky = int(key.split("_y")[1])
                    dist = abs(ky - yaw_int)
                    if dist < min_dist:
                        min_dist = dist
                        closest_yaw_key = key

            if blink_val >= 0.5:
                blink_key = closest_yaw_key if closest_yaw_key else "blink"
            else:
                blink_key = closest_yaw_key.replace("blink", "blink_half") if closest_yaw_key else "blink_half"
                if blink_key not in specials:
                    blink_key = closest_yaw_key if closest_yaw_key else "blink"

            blink_frame = specials.get(blink_key, specials.get("blink"))
            if blink_frame is not None:
                blend = min(1.0, blink_val)
                result = cv2.addWeighted(result, 1 - blend, blink_frame, blend, 0)

        # Blend in smile
        if smile_val > 0.1:
            if smile_val >= 0.5:
                smile_frame = specials.get("smile", specials.get("smile_half"))
            else:
                smile_frame = specials.get("smile_half", specials.get("smile"))

            if smile_frame is not None:
                blend = min(1.0, smile_val * 0.7)
                result = cv2.addWeighted(result, 1 - blend, smile_frame, blend, 0)

        return result

    print("Running! Face animation active.\n")

    while True:
        now = time.time()
        dt = min(now - last_time, 0.1)
        last_time = now

        key = cv2.waitKey(16) & 0xFF

        if key == 27:
            break
        elif key in (ord('a'), ord('A')):
            target_yaw = max(-30, target_yaw - move_speed)
            auto_mode = False
        elif key in (ord('d'), ord('D')):
            target_yaw = min(30, target_yaw + move_speed)
            auto_mode = False
        elif key in (ord('w'), ord('W')):
            target_pitch = max(-20, target_pitch - move_speed)
            auto_mode = False
        elif key in (ord('s'), ord('S')):
            target_pitch = min(20, target_pitch + move_speed)
            auto_mode = False
        elif key in (ord('q'), ord('Q')):
            target_roll = max(-10, target_roll - 1.0)
            auto_mode = False
        elif key in (ord('e'), ord('E')):
            target_roll = min(10, target_roll + 1.0)
            auto_mode = False
        elif key in (ord('b'), ord('B')):
            behavior.force_blink()
        elif key in (ord('n'), ord('N')):
            behavior.smile_target = 0.8 if behavior.smile_target < 0.1 else 0.0
        elif key in (ord('m'), ord('M')):
            pass  # mouth handled differently now
        elif key in (ord('p'), ord('P')):
            auto_mode = not auto_mode
            if auto_mode:
                auto_mover = AutoMovement()
                print("Auto movement: ON")
            else:
                print("Auto movement: OFF")
        elif key in (ord('h'), ord('H')):
            show_hud = not show_hud
        elif key in (ord('r'), ord('R')):
            target_yaw = target_pitch = target_roll = 0
            yaw_smooth.snap(0)
            pitch_smooth.snap(0)
            roll_smooth.snap(0)
            auto_mode = False
            behavior.smile_target = 0.0
        elif key in (ord('+'), ord('=')):
            move_speed = min(8.0, move_speed + 0.5)
            print(f"Move speed: {move_speed}")
        elif key in (ord('-'), ord('_')):
            move_speed = max(0.5, move_speed - 0.5)
            print(f"Move speed: {move_speed}")
        elif key in (ord('g'), ord('G')):
            path = generate_ai_face()
            if path:
                img = cv2.imread(path)
                if img is not None:
                    engine.prepare_source(img)
                    engine._debug_printed = False
                    grid, specials, yaw_vals, pitch_vals = prerender_poses(engine)
                    target_yaw = target_pitch = target_roll = 0
                    yaw_smooth.snap(0); pitch_smooth.snap(0); roll_smooth.snap(0)
        elif key in (ord('l'), ord('L')):
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk(); root.withdraw()
                path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
                root.destroy()
                if path:
                    save_face_to_gallery(path)
                    img = cv2.imread(path)
                    if img is not None:
                        engine.prepare_source(img)
                        engine._debug_printed = False
                        grid, specials, yaw_vals, pitch_vals = prerender_poses(engine)
                        target_yaw = target_pitch = target_roll = 0
                        yaw_smooth.snap(0); pitch_smooth.snap(0); roll_smooth.snap(0)
            except:
                pass

        # Auto movement
        if auto_mode:
            auto_yaw, auto_pitch = auto_mover.update(dt)
            target_yaw = auto_yaw
            target_pitch = auto_pitch

        # Natural behavior (always active)
        nat = behavior.update(dt)

        # Smooth interpolation of head position
        yaw_smooth.set_target(target_yaw + nat["micro_yaw"])
        pitch_smooth.set_target(target_pitch + nat["micro_pitch"])
        roll_smooth.set_target(target_roll + nat["micro_roll"])

        cur_yaw = yaw_smooth.update(dt)
        cur_pitch = pitch_smooth.update(dt)
        cur_roll = roll_smooth.update(dt)

        # Get blended frame
        frame = get_blended_frame(cur_yaw, cur_pitch, nat["blink"], nat["smile"])

        if frame is not None:
            display = frame.copy()

            # Resize if needed
            if display.shape[0] != RES:
                display = cv2.resize(display, (RES, RES), interpolation=cv2.INTER_LINEAR)

            if show_hud:
                mode_str = "AUTO" if auto_mode else "MANUAL"
                info = f"Y:{cur_yaw:.1f} P:{cur_pitch:.1f} R:{cur_roll:.1f} [{mode_str}]"
                cv2.putText(display, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
                if nat["blink"] > 0.1:
                    cv2.putText(display, "BLINK", (RES - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

            cv2.imshow("AI Face Cam", display)

            if vcam is not None:
                try:
                    vcam_frame = display if display.shape[0] == RES else cv2.resize(display, (RES, RES))
                    # Remove HUD for virtual cam output
                    if show_hud:
                        vcam_frame = frame.copy()
                        if vcam_frame.shape[0] != RES:
                            vcam_frame = cv2.resize(vcam_frame, (RES, RES), interpolation=cv2.INTER_LINEAR)
                    vcam.send(vcam_frame)
                except:
                    pass

    if vcam:
        vcam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
