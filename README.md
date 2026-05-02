# Smart Attendance System with Face Recognition and Live Tracking

A complete real-time smart attendance system built with Python and OpenCV that automates attendance tracking using computer vision techniques.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Rubric Mapping](#rubric-mapping)
7. [Keyboard Controls](#keyboard-controls)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Project Title:** Smart Attendance System with Face Recognition and Live Tracking

**Problem Statement:** Traditional attendance systems (manual sign-in or RFID cards) are time-consuming and prone to errors or misuse. This system automates attendance tracking using computer vision, ensuring accuracy, speed, and security.

**Input Source:** Live video stream from webcam or pre-recorded video files

---

## Features

- **Real-time face detection** using Haar Cascades, DNN, or face_recognition library
- **Face recognition** against a stored database of registered users
- **Multi-face tracking** across frames to prevent duplicate entries
- **Liveness detection** via eye detection (prevents photo spoofing)
- **Gesture recognition** - wave to confirm presence
- **Preprocessing pipeline** - resize, blur, grayscale, edge detection, noise reduction
- **Annotated output** with bounding boxes, names, and timestamps
- **Attendance reports** in CSV and Excel formats
- **Daily report generation** with statistics
- **Interactive controls** via keyboard

---

## Installation

### Prerequisites
- Python 3.8+
- Webcam (for live mode)

### Step 1: Clone/Extract the project
```bash
cd smart_attendance_system
```

### Step 2: Install dependencies

**Option A: Full installation (recommended for production)**
```bash
pip install -r requirements.txt
```
> Note: `face_recognition` requires `dlib` which may need CMake. On Windows, use pre-built wheels or WSL.

**Option B: Quick demo (no dlib required)**
```bash
pip install opencv-python numpy pandas openpyxl
```
Then run `python demo_mode.py` instead of `main.py`.

### Step 3: Register faces
```bash
python register_face.py
```

### Step 4: Run the system
```bash
python main.py
```

---

## Usage

### 1. Register Users

**Interactive mode:**
```bash
python register_face.py
```

**Command line:**
```bash
# From camera
python register_face.py --name "John Doe" --id "EMP001"

# From image file
python register_face.py --name "Jane Smith" --image photos/jane.jpg

# List all users
python register_face.py --list

# Delete a user
python register_face.py --delete "John Doe"
```

### 2. Run Attendance System

**Full mode (requires face_recognition library):**
```bash
python main.py
```

**Demo mode (works without face_recognition):**
```bash
python demo_mode.py
```

### 3. View Reports

Reports are saved in the `reports/` directory:
- `attendance_log.csv` - Raw attendance data
- `attendance_report.xlsx` - Formatted Excel report with summary
- `daily_report_YYYY-MM-DD.txt` - Text summary

---

## Project Structure

```
smart_attendance_system/
│
├── main.py                  # Main application (full mode)
├── demo_mode.py             # Demo mode (no dlib required)
├── register_face.py         # Face registration utility
│
├── config.py                # Configuration settings
├── preprocessing.py         # Image preprocessing pipeline
├── detector.py              # Face & eye detection
├── recognizer.py            # Face recognition
├── tracker.py               # Face tracking across frames
├── gesture.py               # Wave gesture detection
├── database_manager.py      # Attendance logging & reports
│
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── database/               # Data storage
│   ├── faces/              # Registered face images
│   └── face_database.pkl   # Face encodings database
│
└── reports/                # Generated reports
    ├── attendance_log.csv
    ├── attendance_report.xlsx
    └── daily_report_*.txt
```

---

## Rubric Mapping

### 1. Project Title
✅ **Smart Attendance System with Face Recognition and Live Tracking**
- Clear and descriptive name defined in `config.py`

### 2. Problem Statement / Objective
✅ **Automated attendance tracking using computer vision**
- Eliminates manual sign-in errors and RFID misuse
- Ensures accuracy, speed, and security
- Defined in `config.py` and documented in README

### 3. Input Data Source
✅ **Multiple input types supported:**
- Live video stream from webcam (`VIDEO_SOURCE = 0`)
- Pre-recorded video files (`VIDEO_SOURCE = "video.mp4"`)
- Image datasets for training (`database/faces/`)
- Configurable in `config.py`

### 4. Preprocessing Steps
✅ **Complete preprocessing pipeline in `preprocessing.py`:**
- **Resize** frames for faster processing
- **Grayscale / HSV** conversion for better detection
- **Gaussian blur** for noise reduction
- **Edge detection** (Canny/Sobel) for feature clarity
- **Noise reduction** (Gaussian, median, bilateral filters)
- All steps configurable via `config.py`

### 5. Feature Extraction / Detection
✅ **Multiple detection methods in `detector.py`:**
- **Face detection:** Haar cascades, DNN, or face_recognition library
- **Eye detection:** For liveness verification (prevents photo spoofing)
- **Contour detection:** For bounding boxes and shape analysis
- Configurable detection method in `config.py`

### 6. Computer Vision Technique
✅ **Two core techniques implemented:**
- **Recognition:** Face identification against stored database (`recognizer.py`)
  - Uses 128-d face encodings (face_recognition) or LBPH fallback
  - Tolerance threshold for matching accuracy
- **Tracking:** Centroid-based tracking across frames (`tracker.py`)
  - Prevents duplicate attendance entries
  - Handles face disappearance and reappearance
  - Configurable max distance and disappeared frames

### 7. Output / Result Display
✅ **Rich output system in `main.py` and `database_manager.py`:**
- **Annotated frames:** Bounding boxes with names, confidence, timestamps
- **Real-time display:** `cv2.imshow()` with live attendance log panel
- **CSV export:** `attendance_log.csv` with all records
- **Excel report:** `attendance_report.xlsx` with summary sheet
- **Daily text report:** Formatted daily summary
- **Auto-save:** Every 5 minutes (configurable)

### 8. User Interaction
✅ **Multiple interaction methods in `main.py`:**
- **Real-time feedback:** "Attendance marked for [Name]" displayed on screen
- **Key presses:**
  - `q` - Quit application
  - `s` - Save report immediately
  - `r` - Register new face
  - `g` - Toggle gesture detection
  - `l` - Toggle liveness verification
- **Gesture interaction:** Wave hand to confirm presence (`gesture.py`)
- **Visual feedback:** Color-coded boxes, status panels, motion indicators

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save attendance report (CSV + Excel) |
| `r` | Enter face registration mode |
| `g` | Toggle wave gesture detection |
| `l` | Toggle liveness (eye) verification |
| `SPACE` | Capture face during registration |
| `ESC` | Cancel registration |

---

## Troubleshooting

### face_recognition not installing
```bash
# On Ubuntu/Debian
sudo apt-get install cmake libdlib-dev

# On macOS
brew install cmake

# Or use demo_mode.py instead (no dlib required)
python demo_mode.py
```

### Camera not detected
- Check if another application is using the camera
- Try changing `VIDEO_SOURCE` in `config.py`:
  - `0` = Default webcam
  - `1` = External webcam
  - `"video.mp4"` = Video file

### No faces detected
- Ensure good lighting conditions
- Adjust `min_size` in `config.py` for smaller/larger faces
- Try different detection methods (haar, dnn, face_recognition)

### Recognition accuracy issues
- Register multiple images per person
- Ensure consistent lighting during registration and recognition
- Adjust `tolerance` in `config.py` (lower = stricter matching)

---

## License

This project is for educational purposes.

## Credits

Built with:
- OpenCV
- face_recognition (dlib)
- NumPy
- Pandas
