# Project Report: Smart Attendance System

## Executive Summary

This project implements a complete **Real-Time Smart Attendance System Using Face Recognition** that addresses all requirements specified in the rubric. The system uses computer vision techniques to automatically detect, recognize, and track faces from live video streams, marking attendance with timestamps and generating reports.

---

## Detailed Rubric Compliance

### 1. Project Title
**Requirement:** Clear and descriptive name of the project.

**Implementation:**
- File: `config.py` (line 15)
- Value: `"Smart Attendance System with Face Recognition and Live Tracking"`
- The title clearly describes: the domain (attendance), the technology (face recognition), and the capability (live tracking).

---

### 2. Problem Statement / Objective
**Requirement:** What problem are you solving with computer vision?

**Implementation:**
- File: `config.py` (documented), `README.md`, `main.py`
- **Problem:** Traditional attendance systems (manual sign-in, RFID cards) are:
  - Time-consuming (queues at entry points)
  - Prone to errors (illegible handwriting, missed entries)
  - Vulnerable to misuse (buddy punching, lost/stolen cards)
- **Solution:** Automated attendance using computer vision:
  - **Accuracy:** Face recognition eliminates human error
  - **Speed:** Real-time processing with no queues
  - **Security:** Liveness detection prevents photo spoofing
  - **Convenience:** Hands-free, automatic marking

---

### 3. Input Data Source
**Requirement:** Type of input used — Image(s), Video file, Live video stream.

**Implementation:**
- File: `config.py` (lines 28-35)
- **Live video stream:** `VIDEO_SOURCE = 0` (default webcam)
  - Configurable to any camera index
  - Adjustable resolution and FPS
- **Video file:** `VIDEO_SOURCE = "test_video.mp4"` (commented option)
  - Supports any OpenCV-compatible video format
- **Image datasets:** `database/faces/` directory
  - Stores registered user images for training
  - Supports JPG, PNG formats
- **Dynamic switching:** Change `VIDEO_SOURCE` in config without code changes

---

### 4. Preprocessing Steps
**Requirement:** Resize, crop, convert to grayscale or HSV, smoothing (Gaussian blur), thresholding/edge detection, noise reduction.

**Implementation:**
- File: `preprocessing.py` — Complete `Preprocessor` class
- **Resize:** `resize()` method (line 52)
  - Reduces frame dimensions for faster processing
  - Configurable width/height in `config.py`
- **Grayscale:** `to_grayscale()` method (line 58)
  - Converts BGR to single-channel for detection algorithms
  - Optional convert-back for display
- **HSV:** `to_hsv()` method (line 64)
  - Color space conversion for skin detection
  - Useful for gesture recognition and color-based features
- **Gaussian Blur:** `gaussian_blur()` method (line 70)
  - Kernel size: (5,5), sigma: 0 (auto-calculated)
  - Reduces high-frequency noise
- **Edge Detection (Canny):** `edge_detection()` method (line 76)
  - Dual-threshold edge detection
  - Returns edge map for feature clarity
- **Sobel (Alternative):** `sobel_edge_detection()` method (line 82)
  - Gradient-based edge detection
- **Noise Reduction:** `reduce_noise()` method (line 88)
  - Three methods: Gaussian, Median, Bilateral filtering
  - Configurable via `config.py`
- **Pipeline:** `process()` method orchestrates all steps in sequence
  - Returns both processed frame and debug visualizations

---

### 5. Feature Extraction / Detection
**Requirement:** Contour detection, shape detection, object detection, face/eye/hand detection.

**Implementation:**
- File: `detector.py` — `FaceDetector` class
- **Face Detection (3 methods):**
  1. **face_recognition library** (HOG/CNN models) — `_detect_faces_fr()`
  2. **Haar Cascades** — `_detect_faces_haar()` using OpenCV's built-in classifier
  3. **DNN (Deep Neural Network)** — `_detect_faces_dnn()` using Caffe model
  - Configurable via `config.py`: `FACE_DETECTION["method"]`
- **Eye Detection:** `detect_eyes()` method
  - Uses Haar cascade for eye detection within face ROI
  - Returns bounding boxes for each detected eye
- **Liveness Verification:** `verify_liveness()` method
  - Prevents spoofing with static photos
  - Requires minimum 2 eyes to consider face "live"
- **Contour Detection:** `get_face_contours()` method
  - Extracts contours from face region using Canny + findContours
  - Useful for shape analysis
- **Bounding Boxes:** All detections return `(x, y, w, h)` format
- **Visualization:** `draw_detections()` draws boxes, confidence scores, and eye regions

---

### 6. Computer Vision Technique
**Requirement:** Tracking (object tracking, motion tracking) and Recognition (face, object).

**Implementation:**

#### A. Recognition
- File: `recognizer.py` — `FaceRecognizer` class
- **Method:** 128-dimensional face encodings using `face_recognition` library
  - Based on dlib's deep metric learning
  - Each face encoded into a 128-d vector
- **Matching:** Euclidean distance comparison with tolerance threshold
  - `RECOGNITION["tolerance"] = 0.6` (configurable)
  - Lower tolerance = stricter matching
- **Database:** Pickle file stores encodings, names, and IDs
- **Fallback:** `SimpleLBPHRecognizer` using OpenCV LBPH for systems without dlib
- **Registration:** `register_face()` captures and stores new face data
- **Identification:** `recognize_face()` returns name, confidence, and ID

#### B. Tracking
- File: `tracker.py` — `CentroidTracker` class
- **Method:** Centroid-based object tracking
  - Computes center point of each face bounding box
  - Matches centroids between frames using Euclidean distance
- **Features:**
  - **Persistent IDs:** Same person maintains same ID across frames
  - **Disappearance handling:** Tracks faces for 30 frames after they leave view
  - **Distance threshold:** Max 100px jump considered same person
  - **Duplicate prevention:** Attendance marked only once per tracking session
- **State management:**
  - `register()` — New face enters frame
  - `update()` — Match existing or create new track
  - `deregister()` — Face gone for too long
  - `mark_attendance()` — Prevent duplicate entries

---

### 7. Output / Result Display
**Requirement:** Annotated frames, cv2.imshow(), save result (image/video) or generate report.

**Implementation:**

#### A. Annotated Frames
- File: `main.py` — `draw_annotations()` method
- **Bounding boxes:** Color-coded (green = known, red = unknown)
- **Name labels:** Displayed above each face with attendance status (✓ or ...)
- **Timestamps:** Current time shown below each face
- **Confidence scores:** Recognition confidence displayed
- **Liveness indicators:** Eye count and live status
- **Object IDs:** Tracking IDs for debugging

#### B. Real-Time Display
- File: `main.py` — `draw_ui()` method
- **Main window:** `cv2.imshow()` with full UI overlay
- **Attendance log panel:** Bottom-left shows last 5 entries
- **Status panel:** Top-right shows system statistics
- **Real-time feedback:** Floating messages for attendance events
- **FPS counter:** Performance monitoring
- **Controls hint:** Bottom bar shows available keys

#### C. Report Generation
- File: `database_manager.py` — `AttendanceManager` class
- **CSV Export:** `save_csv()` — Raw data with all fields
- **Excel Export:** `save_excel()` — Formatted with summary sheet
- **Daily Report:** `generate_daily_report()` — Text format with statistics
- **Auto-save:** Every 5 minutes (configurable)
- **Manual save:** Press 's' key anytime
- **Fields:** Name, ID, First Seen, Last Seen, Status, Confidence, Method, Entries

---

### 8. User Interaction
**Requirement:** Key presses, gestures, or UI elements; real-time feedback on screen.

**Implementation:**

#### A. Key Presses
- File: `main.py` — `handle_key()` method
- `q` — Quit application gracefully
- `s` — Save attendance report (CSV + Excel + daily report)
- `r` — Enter face registration mode
- `g` — Toggle wave gesture detection on/off
- `l` — Toggle liveness verification on/off

#### B. Gesture Interaction
- File: `gesture.py` — `WaveDetector` class
- **Wave detection:** Motion history analysis
  - Tracks motion variance over 10 frames
  - Detects back-and-forth waving pattern
  - Threshold-based activation
- **Confirmation:** Wave to manually confirm attendance
- **Visual feedback:** "WAVE DETECTED!" message on screen
- **Cooldown:** 1-second cooldown between detections

#### C. Real-Time Feedback
- File: `main.py` — `add_log()` and `draw_ui()` methods
- **Attendance messages:** "Attendance marked for [Name]" appears on screen for 3 seconds
- **Status updates:** Cooldown warnings, save confirmations
- **Visual indicators:**
  - Green box = recognized face
  - Red box = unknown face
  - Checkmark = attendance already marked
  - Motion bar = gesture detection activity
- **Registration mode:** Semi-transparent overlay with instructions

---

## Technical Architecture

```
Video Input → Preprocessing → Face Detection → Recognition → Tracking → Attendance → Reports
                ↓                ↓                ↓            ↓           ↓
            Resize          Haar/DNN       128-d enc    Centroid    CSV/Excel
            Grayscale       Eye detect     LBPH fallback  IDs       Daily report
            Blur            Contours
            Edge detect
            Noise reduction
```

## File Structure Summary

| File | Purpose | Rubric Sections |
|------|---------|-----------------|
| `config.py` | Settings & constants | All |
| `preprocessing.py` | Image preprocessing | 4 |
| `detector.py` | Face/eye detection | 5 |
| `recognizer.py` | Face recognition | 6 (Recognition) |
| `tracker.py` | Face tracking | 6 (Tracking) |
| `gesture.py` | Wave detection | 8 |
| `database_manager.py` | Logging & reports | 7 |
| `main.py` | Main application | All |
| `demo_mode.py` | Standalone demo | All (fallback) |
| `register_face.py` | User registration | 3, 6 |
| `test_system.py` | Validation tests | All |

## Conclusion

This project fully implements all 8 rubric requirements with a modular, extensible architecture. Each component is independently testable and configurable. The system supports both production deployment (with face_recognition library) and quick demonstrations (with OpenCV-only demo mode).
