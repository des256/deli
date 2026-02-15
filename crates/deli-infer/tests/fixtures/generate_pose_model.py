#!/usr/bin/env python3
"""
Download YOLO26n-pose ONNX models and test image from HuggingFace.

Usage:
    python generate_pose_model.py

Requirements:
    pip install huggingface_hub requests
"""

import os
import sys
from pathlib import Path

try:
    import requests
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install huggingface_hub requests")
    sys.exit(1)

FIXTURES_DIR = Path(__file__).parent
REPO_ID = "onnx-community/yolo26n-pose-ONNX"

MODELS = [
    ("onnx/model.onnx", "yolo26n-pose.onnx"),
    ("onnx/model_uint8.onnx", "yolo26n-pose-uint8.onnx"),
]

TEST_IMAGE_URL = "https://cdn.create.vista.com/api/media/small/44316099/stock-photo-happy-family-have-fun-walking-on-beach-at-sunset"
TEST_IMAGE_PATH = "test_pose_image.jpg"


def download_model(repo_id, filename, output_path):
    """Download a model file from HuggingFace."""
    print(f"Downloading {filename}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=None
        )

        with open(downloaded_path, "rb") as src:
            with open(output_path, "wb") as dst:
                dst.write(src.read())

        file_size = output_path.stat().st_size
        if file_size == 0:
            print(f"Error: Downloaded file {output_path} is empty")
            return False

        print(f"  ✓ Downloaded {output_path.name} ({file_size / 1024 / 1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {filename}: {e}")
        return False


def download_test_image(url, output_path):
    """Download test image from URL."""
    print(f"Downloading test image...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        file_size = output_path.stat().st_size
        if file_size == 0:
            print(f"Error: Downloaded image {output_path} is empty")
            return False

        print(f"  ✓ Downloaded {output_path.name} ({file_size / 1024:.2f} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download test image: {e}")
        return False


def main():
    """Download all required fixtures."""
    print("YOLO26n-pose Model Downloader")
    print("=" * 50)

    success_count = 0
    total_count = len(MODELS) + 1

    for repo_filename, local_filename in MODELS:
        output_path = FIXTURES_DIR / local_filename
        if output_path.exists():
            print(f"Skipping {local_filename} (already exists)")
            success_count += 1
        elif download_model(REPO_ID, repo_filename, output_path):
            success_count += 1

    image_path = FIXTURES_DIR / TEST_IMAGE_PATH
    if image_path.exists():
        print(f"Skipping {TEST_IMAGE_PATH} (already exists)")
        success_count += 1
    elif download_test_image(TEST_IMAGE_URL, image_path):
        success_count += 1

    print("=" * 50)
    if success_count == total_count:
        print(f"✓ All {total_count} files ready!")
        print("\nYou can now run the integration test:")
        print(
            "  cargo test -p deli-infer --features onnx --test pose_integration_tests"
        )
    else:
        print(f"✗ {total_count - success_count} files failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()
