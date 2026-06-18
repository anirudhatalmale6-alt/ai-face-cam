"""Download required InsightFace models for AI Face Cam 3D build"""
import os
import sys
import zipfile
import urllib.request
import shutil

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
NEEDED_FILES = ["det_10g.onnx", "2d106det.onnx"]


def download_from_insightface_release():
    """Download buffalo_l.zip from InsightFace GitHub releases and extract needed models"""
    zip_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    zip_path = os.path.join(MODELS_DIR, "buffalo_l.zip")
    dest_dir = os.path.join(MODELS_DIR, "buffalo_l")

    os.makedirs(dest_dir, exist_ok=True)

    all_exist = all(os.path.exists(os.path.join(dest_dir, f)) for f in NEEDED_FILES)
    if all_exist:
        print("All models already exist!")
        return True

    print(f"Downloading buffalo_l.zip from InsightFace releases (~300MB)...")

    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = int(count * block_size * 100 / total_size)
            mb = count * block_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct}% ({mb:.0f}MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(zip_url, zip_path, reporthook=progress)
        print("\n  Download complete!")
    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False

    print("  Extracting needed models...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                basename = os.path.basename(name)
                if basename in NEEDED_FILES:
                    data = zf.read(name)
                    dest = os.path.join(dest_dir, basename)
                    with open(dest, 'wb') as f:
                        f.write(data)
                    print(f"  Extracted: {basename} ({len(data) / (1024*1024):.1f}MB)")
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("  Cleaned up zip file")

    return all(os.path.exists(os.path.join(dest_dir, f)) for f in NEEDED_FILES)


def download_via_insightface_lib():
    """Use InsightFace library's built-in download"""
    try:
        from insightface.app import FaceAnalysis
        print("Using InsightFace library to download models...")
        app = FaceAnalysis(
            name="buffalo_l",
            root=os.path.dirname(MODELS_DIR),
            providers=['CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("Models downloaded via InsightFace!")
        return True
    except Exception as e:
        print(f"InsightFace download failed: {e}")
        return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Preparing face models for build...")

    dest_dir = os.path.join(MODELS_DIR, "buffalo_l")
    all_exist = all(os.path.exists(os.path.join(dest_dir, f)) for f in NEEDED_FILES)
    if all_exist:
        print("All models already present!")
        return

    if download_via_insightface_lib():
        return

    if download_from_insightface_release():
        return

    print("ERROR: Could not download models!")
    sys.exit(1)


if __name__ == "__main__":
    main()
