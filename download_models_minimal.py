"""Download only the required models for AI Face Cam 3D build"""
import os
import urllib.request
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "det_10g": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/det_10g.onnx",
        "filename": "buffalo_l/det_10g.onnx",
        "size_mb": 16
    },
    "2d106det": {
        "url": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/2d106det.onnx",
        "filename": "buffalo_l/2d106det.onnx",
        "size_mb": 5
    },
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Downloading face models for build...")

    for name, info in MODELS.items():
        dest = os.path.join(MODELS_DIR, info["filename"])
        if os.path.exists(dest):
            print(f"  Already exists: {dest}")
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"  Downloading {name} (~{info['size_mb']}MB)...")

        def progress(count, block_size, total_size):
            pct = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(info["url"], dest, reporthook=progress)
        print(f"\r  Done: {dest}")

    print("\nModels ready!")


if __name__ == "__main__":
    main()
