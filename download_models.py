"""Download required face models for AI Face Cam"""
import os
import urllib.request
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "buffalo_l": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
        "filename": "inswapper_128.onnx",
        "size_mb": 555
    },
    "det_10g": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/det_10g.onnx",
        "filename": "buffalo_l/det_10g.onnx",
        "size_mb": 16
    },
    "w600k_r50": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/w600k_r50.onnx",
        "filename": "buffalo_l/w600k_r50.onnx",
        "size_mb": 174
    },
    "2d106det": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/2d106det.onnx",
        "filename": "buffalo_l/2d106det.onnx",
        "size_mb": 5
    },
    "1k3d68": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/1k3d68.onnx",
        "filename": "buffalo_l/1k3d68.onnx",
        "size_mb": 143
    },
    "genderage": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/genderage.onnx",
        "filename": "buffalo_l/genderage.onnx",
        "size_mb": 1
    },
}


def download_file(url, dest, desc=""):
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {desc}...")

    def progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r  {pct}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\r  Done: {dest}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Downloading face models...")

    for name, info in MODELS.items():
        dest = os.path.join(MODELS_DIR, info["filename"])
        download_file(info["url"], dest, f"{name} (~{info['size_mb']}MB)")

    print("\nAll models downloaded!")


if __name__ == "__main__":
    main()
