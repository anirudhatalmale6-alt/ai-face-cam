"""
AI Face Cam - LivePortrait ONNX
Real-time face animation using LivePortrait ONNX models.
No PyTorch needed - uses onnxruntime + numpy only.
"""
import os
import sys
import time
import urllib.request
import numpy as np
import cv2

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

MASK_CROP = None


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
    """Download a random AI-generated face"""
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


class OneEuroFilter:
    def __init__(self, t_e=1.0, alpha=0.3):
        self.t_e = t_e
        self.alpha = alpha
        self.prev = None

    def __call__(self, x):
        if self.prev is None:
            self.prev = x.copy()
            return x
        result = self.alpha * x + (1 - self.alpha) * self.prev
        self.prev = result.copy()
        return result


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

        if plugin_dll:
            try:
                opts.register_custom_ops_library(plugin_dll)
                print(f"  Registered GridSample3D custom op")
            except Exception as e:
                print(f"  Warning: Could not register custom op: {e}")
        else:
            print("  Warning: grid_sample_3d_ort.dll not found!")

        for name in MODEL_FILES:
            if name.endswith(".dll"):
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
            print("  Warping model outputs:")
            for out in warp_sess.get_outputs():
                print(f"    {out.name}: {out.shape} ({out.type})")
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
        """Initialize OpenCV DNN face detector"""
        if not hasattr(self, '_face_det'):
            proto = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self._face_cascade = cv2.CascadeClassifier(proto)
            try:
                self._face_det = cv2.FaceDetectorYN.create(
                    "", "", (300, 300), 0.5, 0.3, 5000
                )
                self._use_dnn = False
            except:
                self._use_dnn = False

    def detect_face(self, img_bgr):
        """Detect face and return 106 landmarks"""
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
        """Crop face region centered on landmarks"""
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
        """Process source image - extract features and motion"""
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
        crop_512, M_512, M_512_inv = self.crop_face(img_rgb, lmk, target_size=512)

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
            "M_512_inv": M_512_inv,
            "mask_crop": mask_crop,
            "orig_shape": (h, w),
            "lmk": lmk,
            "eye_ratio": src_eye_ratio,
            "lip_ratio": src_lip_ratio,
        }

        print(f"  Source ready! Keypoints: {num_kp}, Pose: pitch={pitch[0]:.1f} yaw={yaw[0]:.1f} roll={roll[0]:.1f}")
        return True

    def animate_frame(self, driving_img_bgr, d0_info=None):
        """Animate source face using driving frame's motion"""
        if self.src_info is None:
            return None

        driving_rgb = cv2.cvtColor(driving_img_bgr, cv2.COLOR_BGR2RGB)
        lmk_d = self.detect_face(driving_img_bgr)
        if lmk_d is None:
            return None

        crop_d, _, _ = self.crop_face(driving_rgb, lmk_d)
        crop_input = (crop_d.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        mot_out = self._run_model("motion_extractor", crop_input)
        pitch_d, yaw_d, roll_d, t_d, exp_d, scale_d, kp_d = mot_out
        pitch_d = headpose_pred_to_degree(pitch_d)
        yaw_d = headpose_pred_to_degree(yaw_d)
        roll_d = headpose_pred_to_degree(roll_d)
        exp_d = exp_d.reshape(1, -1, 3)
        R_d = get_rotation_matrix(pitch_d, yaw_d, roll_d)

        info = {
            "pitch": pitch_d, "yaw": yaw_d, "roll": roll_d,
            "t": t_d, "exp": exp_d, "scale": scale_d,
            "R": R_d,
        }

        if d0_info is None:
            return info

        src = self.src_info
        R_new = R_d @ np.linalg.inv(d0_info["R"]) @ src["R_s"]
        delta_exp = src["exp"] + (exp_d - d0_info["exp"])
        scale_new = src["scale"] * (scale_d / (d0_info["scale"] + 1e-6))
        t_new = src["t"] + (t_d - d0_info["t"])
        t_new[..., 2] = 0

        x_d_new = scale_new[..., None] * (src["kp"] @ R_new + delta_exp) + t_new[:, None, :]

        feat_stitch = np.concatenate([
            src["x_s"].reshape(1, -1),
            x_d_new.reshape(1, -1)
        ], axis=1).astype(np.float32)
        delta = self._run_model("stitching", feat_stitch)[0]
        num_kp = src["num_kp"]
        x_d_new = x_d_new + delta[..., :3 * num_kp].reshape(1, num_kp, 3)
        x_d_new[..., :2] += delta[..., 3 * num_kp:3 * num_kp + 2].reshape(1, 1, 2)

        out = self._run_model(
            "warping_spade-fix",
            src["f_s"],
            x_d_new.astype(np.float32),
            src["x_s"].astype(np.float32)
        )[0]

        out_t = out[0].transpose(1, 2, 0)
        out_img = np.clip(out_t, 0, 1) * 255
        out_img = out_img.astype(np.uint8)

        return out_img

    def animate_keyboard(self, pitch_delta=0, yaw_delta=0, roll_delta=0, exp_scale=1.0, blink=0, smile=0, mouth=0):
        """Animate source face using keyboard-controlled motion parameters"""
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


def main():
    print("=" * 50)
    print("  AI Face Cam - LivePortrait ONNX")
    print("  Real-time face animation")
    print("=" * 50)
    print()

    if not download_all_models():
        input("Press Enter to exit...")
        return

    engine = LivePortraitEngine()
    engine.load_models()

    source_path = None
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            source_path = arg
            break

    if source_path is None:
        print("No source face provided. Options:")
        print("  G = Generate random AI face")
        print("  C = Choose from gallery (18 AI faces, 3 pages)")
        print("  L = Load a photo from file")
        print()
        choice = input("Choose (G/C/L): ").strip().upper()
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
            except:
                source_path = input("Enter path to face photo: ").strip().strip('"')
        elif choice == "C":
            num_faces = 18
            print(f"Generating {num_faces} AI faces (this takes about 30 seconds)...")
            face_paths = []
            for i in range(num_faces):
                sys.stdout.write(f"\r  Downloading face {i+1}/{num_faces}...")
                sys.stdout.flush()
                p = generate_ai_face()
                if p:
                    face_paths.append(p)
            print(f"\n  {len(face_paths)} faces ready!\n")
            if face_paths:
                per_page = 6
                pages = (len(face_paths) + per_page - 1) // per_page
                page = 0
                thumbs = []
                for p in face_paths:
                    t = cv2.imread(p)
                    if t is not None:
                        thumbs.append(cv2.resize(t, (170, 170)))
                    else:
                        thumbs.append(np.zeros((170, 170, 3), dtype=np.uint8))
                while True:
                    gallery = np.zeros((420, 530, 3), dtype=np.uint8)
                    start = page * per_page
                    end = min(start + per_page, len(thumbs))
                    cv2.putText(gallery, f"Page {page+1}/{pages} - Press 1-6 to pick, A/D for pages", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    for i in range(start, end):
                        idx = i - start
                        row, col = idx // 3, idx % 3
                        y0, x0 = 35 + row * 185, 5 + col * 175
                        gallery[y0:y0+170, x0:x0+170] = thumbs[i]
                        cv2.putText(gallery, str(idx+1), (x0+5, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("AI Face Cam", gallery)
                    k = cv2.waitKey(100) & 0xFF
                    if k >= ord('1') and k <= ord('6'):
                        sel = start + (k - ord('1'))
                        if sel < len(face_paths):
                            source_path = face_paths[sel]
                            print(f"  Selected face #{sel+1}")
                            break
                    elif k == ord('d') or k == ord('D') or k == 83:
                        page = min(pages - 1, page + 1)
                    elif k == ord('a') or k == ord('A') or k == 81:
                        page = max(0, page - 1)
                    elif k == 27:
                        source_path = face_paths[0]
                        break
            else:
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
    if HAS_VCAM:
        try:
            vcam = pyvirtualcam.Camera(width=512, height=512, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Virtual camera: {vcam.device}")
        except Exception as e:
            print(f"Virtual camera not available: {e}")
            print("Install OBS Studio for virtual camera output.")

    def prerender_poses(eng):
        yaw_vals = np.arange(-30, 31, 3)
        pitch_vals = np.arange(-20, 21, 4)
        grid = {}
        specials = {}
        total = len(yaw_vals) * len(pitch_vals) + 4
        count = 0
        print(f"\nPre-rendering {total} pose frames (takes a few minutes)...")
        print("After this, the face moves INSTANTLY with keyboard.\n")
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
        for label, kwargs in [("blink", {"blink": 1.0}), ("smile", {"smile": 1.0}), ("mouth", {"mouth": 1.0}), ("blink+smile", {"blink": 1.0, "smile": 1.0})]:
            out = eng.animate_keyboard(0, 0, 0, **kwargs)
            if out is not None:
                specials[label] = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            count += 1
            sys.stdout.write(f"\r  [{('#' * (count*100//total//5))}{('-' * (20-count*100//total//5))}] {count*100//total}% ({count}/{total})")
            sys.stdout.flush()
        print(f"\n  Done! {len(grid)} pose frames + {len(specials)} expressions ready.\n")
        return grid, specials, yaw_vals, pitch_vals

    grid, specials, yaw_vals, pitch_vals = prerender_poses(engine)

    print("Controls (INSTANT - no delay!):")
    print("  A/D  = Turn head left/right")
    print("  W/S  = Look up/down")
    print("  B    = Blink")
    print("  N    = Smile")
    print("  M    = Open mouth")
    print("  R    = Reset to center")
    print("  P    = Play auto-loop (for KYC)")
    print("  G    = New AI face")
    print("  L    = Load photo")
    print("  ESC  = Quit")
    print()

    cur_yaw_idx = len(yaw_vals) // 2
    cur_pitch_idx = len(pitch_vals) // 2
    expression = None
    auto_loop = False
    loop_seq = []
    for y in range(len(yaw_vals)//2, -1, -1):
        loop_seq.append((len(pitch_vals)//2, y))
    for y in range(0, len(yaw_vals)):
        loop_seq.append((len(pitch_vals)//2, y))
    for y in range(len(yaw_vals)-1, len(yaw_vals)//2, -1):
        loop_seq.append((len(pitch_vals)//2, y))
    for p in range(len(pitch_vals)//2, 0, -1):
        loop_seq.append((p, len(yaw_vals)//2))
    for p in range(0, len(pitch_vals)):
        loop_seq.append((p, len(yaw_vals)//2))
    for p in range(len(pitch_vals)-1, len(pitch_vals)//2, -1):
        loop_seq.append((p, len(yaw_vals)//2))
    loop_seq.append((len(pitch_vals)//2, len(yaw_vals)//2))
    loop_seq.append((len(pitch_vals)//2, len(yaw_vals)//2))
    loop_idx = 0
    blink_timer = 0

    def get_frame(pi, yi, expr=None):
        p = int(pitch_vals[np.clip(pi, 0, len(pitch_vals)-1)])
        y = int(yaw_vals[np.clip(yi, 0, len(yaw_vals)-1)])
        if expr and expr in specials:
            return specials[expr]
        return grid.get((p, y), grid.get((0, 0)))

    while True:
        key = cv2.waitKey(33) & 0xFF

        if key == 27:
            break
        elif key == ord('p') or key == ord('P'):
            auto_loop = not auto_loop
            print(f"Auto-loop: {'ON' if auto_loop else 'OFF'}")
        elif key == ord('a') or key == ord('A'):
            cur_yaw_idx = max(0, cur_yaw_idx - 1); auto_loop = False; expression = None
        elif key == ord('d') or key == ord('D'):
            cur_yaw_idx = min(len(yaw_vals)-1, cur_yaw_idx + 1); auto_loop = False; expression = None
        elif key == ord('w') or key == ord('W'):
            cur_pitch_idx = max(0, cur_pitch_idx - 1); auto_loop = False; expression = None
        elif key == ord('s') or key == ord('S'):
            cur_pitch_idx = min(len(pitch_vals)-1, cur_pitch_idx + 1); auto_loop = False; expression = None
        elif key == ord('b') or key == ord('B'):
            expression = "blink" if expression != "blink" else None
        elif key == ord('n') or key == ord('N'):
            expression = "smile" if expression != "smile" else None
        elif key == ord('m') or key == ord('M'):
            expression = "mouth" if expression != "mouth" else None
        elif key == ord('r') or key == ord('R'):
            cur_yaw_idx = len(yaw_vals) // 2; cur_pitch_idx = len(pitch_vals) // 2
            expression = None; auto_loop = False
        elif key == ord('g') or key == ord('G'):
            path = generate_ai_face()
            if path:
                img = cv2.imread(path)
                if img is not None:
                    engine.prepare_source(img)
                    engine._debug_printed = False
                    grid, specials, yaw_vals, pitch_vals = prerender_poses(engine)
                    cur_yaw_idx = len(yaw_vals) // 2
                    cur_pitch_idx = len(pitch_vals) // 2
        elif key == ord('l') or key == ord('L'):
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk(); root.withdraw()
                path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
                root.destroy()
                if path:
                    img = cv2.imread(path)
                    if img is not None:
                        engine.prepare_source(img)
                        engine._debug_printed = False
                        grid, specials, yaw_vals, pitch_vals = prerender_poses(engine)
                        cur_yaw_idx = len(yaw_vals) // 2
                        cur_pitch_idx = len(pitch_vals) // 2
            except:
                pass

        if auto_loop:
            blink_timer += 1
            if blink_timer == 90:
                expression = "blink"
            elif blink_timer == 94:
                expression = None
            elif blink_timer > 200:
                blink_timer = 0

            cur_pitch_idx, cur_yaw_idx = loop_seq[loop_idx % len(loop_seq)]
            loop_idx += 1

        frame = get_frame(cur_pitch_idx, cur_yaw_idx, expression)
        if frame is not None:
            display = frame.copy()
            p_val = pitch_vals[np.clip(cur_pitch_idx, 0, len(pitch_vals)-1)]
            y_val = yaw_vals[np.clip(cur_yaw_idx, 0, len(yaw_vals)-1)]
            mode = "AUTO-LOOP" if auto_loop else "MANUAL"
            info = f"Yaw:{y_val:.0f} Pitch:{p_val:.0f} [{mode}]"
            cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("AI Face Cam", display)

            if vcam is not None:
                try:
                    vcam_frame = cv2.resize(frame, (512, 512))
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
