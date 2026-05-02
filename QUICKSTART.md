# Quick Start Guide

## 5-Minute Setup

### 1. Install (30 seconds)
```bash
cd smart_attendance_system
python setup.py --minimal
```

### 2. Run Demo (no registration needed)
```bash
python demo_mode.py
```
The demo auto-registers faces and tracks attendance.

### 3. Register Users (2 minutes)
```bash
python register_face.py
```
Follow prompts to capture faces from camera.

### 4. Run Full System
```bash
python main.py
```

### 5. View Reports
Check the `reports/` folder for CSV and Excel files.

---

## Common Issues

| Problem | Solution |
|---------|----------|
| `dlib` won't install | Use `python demo_mode.py` instead |
| No camera | Edit `config.py` and set `VIDEO_SOURCE = "video.mp4"` |
| Faces not detected | Improve lighting; move closer to camera |
| Wrong recognition | Re-register with better lighting |
