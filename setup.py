"""
Setup Script for Smart Attendance System
Handles installation and initial configuration.
"""
import os
import sys
import subprocess
import argparse


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8+ required.")
        sys.exit(1)
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")


def install_dependencies(full=True):
    """Install required packages."""
    print("\n[SETUP] Installing dependencies...")

    if full:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                 "-r", "requirements.txt"])
            print("[OK] All dependencies installed.")
        except subprocess.CalledProcessError:
            print("[WARNING] Full installation failed. Trying minimal...")
            install_dependencies(full=False)
    else:
        minimal = ["opencv-python", "numpy", "pandas", "openpyxl"]
        for pkg in minimal:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"[OK] Installed {pkg}")
            except:
                print(f"[ERROR] Failed to install {pkg}")


def create_directories():
    """Create project directory structure."""
    dirs = ["database/faces", "reports", "models", "test_data"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Directory structure created.")


def download_models():
    """Download pre-trained DNN models if needed."""
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # DNN face detector files
    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    proto_file = os.path.join(model_dir, "deploy.prototxt")
    model_file = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(proto_file):
        print("[SETUP] Downloading DNN deploy.prototxt...")
        try:
            import urllib.request
            urllib.request.urlretrieve(proto_url, proto_file)
            print("[OK] deploy.prototxt downloaded.")
        except:
            print("[WARNING] Could not download deploy.prototxt. Haar cascades will be used.")

    if not os.path.exists(model_file):
        print("[SETUP] Downloading DNN model (87MB)...")
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, model_file)
            print("[OK] DNN model downloaded.")
        except:
            print("[WARNING] Could not download DNN model. Haar cascades will be used.")


def test_installation():
    """Test if core components work."""
    print("\n[SETUP] Testing installation...")

    tests = []

    # Test OpenCV
    try:
        import cv2
        tests.append(("OpenCV", cv2.__version__, True))
    except:
        tests.append(("OpenCV", "Not installed", False))

    # Test NumPy
    try:
        import numpy as np
        tests.append(("NumPy", np.__version__, True))
    except:
        tests.append(("NumPy", "Not installed", False))

    # Test face_recognition
    try:
        import face_recognition
        tests.append(("face_recognition", "Installed", True))
    except:
        tests.append(("face_recognition", "Not installed (use demo_mode.py)", False))

    # Test pandas
    try:
        import pandas as pd
        tests.append(("Pandas", pd.__version__, True))
    except:
        tests.append(("Pandas", "Not installed", False))

    print("\n" + "-" * 50)
    print(f"  {'Package':<20} {'Version/Status':<25} {'OK'}")
    print("-" * 50)
    for name, version, ok in tests:
        status = "✓" if ok else "✗"
        print(f"  {name:<20} {version:<25} {status}")
    print("-" * 50)

    # Camera test
    print("\n[SETUP] Testing camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"[OK] Camera working. Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("[WARNING] Camera opened but cannot read frames.")
    else:
        print("[WARNING] No camera detected. You can still use video files.")
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Setup Smart Attendance System")
    parser.add_argument("--minimal", "-m", action="store_true", 
                       help="Minimal install (no face_recognition)")
    parser.add_argument("--skip-models", "-s", action="store_true",
                       help="Skip downloading DNN models")

    args = parser.parse_args()

    print("=" * 60)
    print("  SMART ATTENDANCE SYSTEM - SETUP")
    print("=" * 60)

    check_python_version()
    create_directories()

    if not args.skip_models:
        download_models()

    install_dependencies(full=not args.minimal)
    test_installation()

    print("\n" + "=" * 60)
    print("  SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Register faces: python register_face.py")
    print("  2. Run system:     python main.py")
    if args.minimal:
        print("  2. Run demo:       python demo_mode.py")
    print()


if __name__ == "__main__":
    main()
