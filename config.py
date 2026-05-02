"""
Configuration Module for Smart Attendance System
Addresses: Project setup and global parameters
"""
import os
import cv2

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
FACES_DIR = os.path.join(DATABASE_DIR, "faces")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
for directory in [DATABASE_DIR, FACES_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# 1. PROJECT TITLE
# =============================================================================
PROJECT_TITLE = "Smart Attendance System with Face Recognition and Live Tracking"
PROJECT_VERSION = "1.0.0"

# =============================================================================
# 3. INPUT DATA SOURCE CONFIGURATION
# =============================================================================
# Video source: 0 for default webcam, or path to video file
VIDEO_SOURCE = 0  # Live video stream from camera
# VIDEO_SOURCE = "test_video.mp4"  # Uncomment to use pre-recorded video

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# =============================================================================
# 4. PREPROCESSING CONFIGURATION
# =============================================================================
PREPROCESSING = {
    "resize": {
        "enabled": False,
        "width": 1280,
        "height": 720
    },
    "grayscale": {
        "enabled": False,  # Set True if using Haar-only pipeline
        "convert_back": True  # Convert back to BGR after processing
    },
    "hsv": {
        "enabled": False,  # Set True for HSV-based skin detection
        "convert_back": True
    },
    "gaussian_blur": {
        "enabled": True,
        "kernel_size": (5, 5),
        "sigma": 0
    },
    "edge_detection": {
        "enabled": False,  # Enable for Canny edge visualization
        "threshold1": 100,
        "threshold2": 200,
        "convert_back": True
    },
    "noise_reduction": {
        "enabled": True,
        "method": "gaussian"  # Options: "gaussian", "median", "bilateral"
    }
}

# =============================================================================
# 5. FEATURE EXTRACTION / DETECTION CONFIGURATION
# =============================================================================
FACE_DETECTION = {
    "method": "face_recognition",  # Options: "face_recognition", "haar", "dnn"
    "haar_cascade_path": cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    "eye_cascade_path": cv2.data.haarcascades + "haarcascade_eye.xml",
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_size": (80, 80)
}

# Eye detection for liveness verification
EYE_DETECTION = {
    "enabled": True,
    "min_eyes": 2,  # Minimum eyes detected to consider face "live"
    "cascade_path": cv2.data.haarcascades + "haarcascade_eye.xml"
}

# =============================================================================
# 6. COMPUTER VISION TECHNIQUE CONFIGURATION
# =============================================================================
RECOGNITION = {
    "tolerance": 0.6,  # Face recognition distance threshold (lower = stricter)
    "model": "hog",    # Options: "hog" (CPU), "cnn" (GPU)
    "upsample": 1      # Times to upsample image for detection
}

TRACKING = {
    "enabled": True,
    "max_disappeared": 30,  # Frames to keep tracking after face disappears
    "max_distance": 100,    # Max pixel distance for same-face matching
    "attendance_cooldown": 60  # Seconds before same person can be marked again
}

# =============================================================================
# 7. OUTPUT / RESULT DISPLAY CONFIGURATION
# =============================================================================
OUTPUT = {
    "show_annotations": True,
    "show_names": True,
    "show_timestamps": True,
    "show_confidence": True,
    "box_color": (0, 255, 0),      # Green bounding box
    "text_color": (255, 255, 255),  # White text
    "unknown_color": (0, 0, 255),   # Red for unknown faces
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.6,
    "thickness": 2
}

REPORT = {
    "auto_save_interval": 300,  # Auto-save every 5 minutes (seconds)
    "csv_filename": "attendance_log.csv",
    "excel_filename": "attendance_report.xlsx",
    "daily_report": True
}

# =============================================================================
# 8. USER INTERACTION CONFIGURATION
# =============================================================================
INTERACTION = {
    "gesture_enabled": True,      # Wave to confirm presence
    "wave_threshold": 15,         # Motion threshold for wave detection
    "wave_history": 10,           # Frames of motion history
    "key_quit": ord('q'),
    "key_save": ord('s'),
    "key_register": ord('r'),
    "key_toggle_gesture": ord('g'),
    "key_toggle_liveness": ord('l')
}

# Display settings
DISPLAY = {
    "window_name": PROJECT_TITLE,
    "show_fps": True,
    "show_attendance_log": True,
    "max_log_entries": 5,
    "log_display_duration": 3  # Seconds to show attendance message
}
