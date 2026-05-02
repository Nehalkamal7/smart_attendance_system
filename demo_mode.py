"""
Demo Mode - Standalone Attendance System
Works without face_recognition library using OpenCV built-in methods.
Use this for quick testing and demonstrations.

Usage:
    python demo_mode.py
"""
import cv2
import numpy as np
import time
from datetime import datetime
import os
import csv

from config import (
    VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, 
    OUTPUT, DISPLAY, INTERACTION, REPORTS_DIR
)
from preprocessing import Preprocessor
from detector import FaceDetector
from tracker import CentroidTracker
from gesture import WaveDetector


class DemoAttendanceSystem:
    """Demo version using OpenCV Haar cascades (no face_recognition needed)."""

    def __init__(self):
        print("=" * 70)
        print("  SMART ATTENDANCE SYSTEM - DEMO MODE")
        print("  (Using OpenCV Haar Cascades - No dlib required)")
        print("=" * 70)

        self.preprocessor = Preprocessor()
        self.detector = FaceDetector()
        self.tracker = CentroidTracker()
        self.gesture = WaveDetector()

        # Demo: Simulate known users with face templates
        self.known_faces = {}  # name -> (template, id)
        self.next_id = 1

        self.gesture_enabled = True
        self.liveness_enabled = True
        self.running = True

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.log_messages = []

        # Attendance records
        self.attendance_records = {}
        self.csv_path = os.path.join(REPORTS_DIR, "demo_attendance.csv")

        self._init_video()
        print("[INFO] Demo mode ready. Press 'q' to quit.")
        print("-" * 70)

    def _init_video(self):
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        if isinstance(VIDEO_SOURCE, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video source")

    def add_log(self, msg):
        self.log_messages.append((msg, time.time()))
        self.log_messages = [(m, t) for m, t in self.log_messages 
                            if time.time() - t < 3]

    def recognize_face_demo(self, face_roi):
        """
        Simple demo recognition using LBPH or template matching.
        In demo mode, we use a simple approach:
        - If face matches a known template, return name
        - Otherwise, auto-register as "Person_X"
        """
        if face_roi.size == 0:
            return "Unknown", 0.0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))

        best_match = None
        best_score = float('inf')

        for name, (template, _) in self.known_faces.items():
            # Template matching using normalized correlation
            if template.shape == gray.shape:
                diff = cv2.norm(gray.astype(np.float32), template.astype(np.float32), cv2.NORM_L2)
                if diff < best_score:
                    best_score = diff
                    best_match = name

        # Threshold for recognition
        if best_match and best_score < 5000:
            confidence = max(0, 1 - (best_score / 10000))
            return best_match, confidence

        # Auto-register new face
        new_name = f"Person_{self.next_id:03d}"
        self.known_faces[new_name] = (gray.copy(), f"DEMO_{self.next_id:03d}")
        self.next_id += 1

        return new_name, 0.5

    def mark_attendance(self, name, person_id):
        """Mark attendance for demo."""
        now = datetime.now()
        if name not in self.attendance_records:
            self.attendance_records[name] = {
                "name": name,
                "id": person_id,
                "first_seen": now,
                "last_seen": now,
                "entries": 1
            }
            return True, f"Attendance marked for {name}"
        else:
            self.attendance_records[name]["last_seen"] = now
            self.attendance_records[name]["entries"] += 1
            return False, f"Already marked: {name}"

    def save_csv(self):
        """Save demo attendance to CSV."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "ID", "First Seen", "Last Seen", "Entries"])
            for r in self.attendance_records.values():
                writer.writerow([
                    r["name"], r["id"],
                    r["first_seen"].strftime("%H:%M:%S"),
                    r["last_seen"].strftime("%H:%M:%S"),
                    r["entries"]
                ])
        print(f"[SAVE] Demo attendance saved: {self.csv_path}")

    def draw_ui(self, frame, faces, tracked):
        """Draw all UI elements."""
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Draw face boxes and labels
        for face, (oid, centroid, name, is_new, marked) in zip(faces, tracked):
            x, y, fw, fh = face["bbox"]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            status = "✓" if marked else ""

            cv2.rectangle(annotated, (x, y), (x + fw, y + fh), color, 2)
            label = f"{name} {status}"
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Liveness
            if self.liveness_enabled:
                roi = frame[y:y+fh, x:x+fw]
                if roi.size > 0:
                    is_live, count, eyes = self.detector.verify_liveness(roi)
                    live_color = (0, 255, 0) if is_live else (0, 0, 255)
                    cv2.putText(annotated, f"Eyes: {count}", (x, y + fh + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, live_color, 1)

        # Header
        cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(annotated, "Smart Attendance - DEMO MODE", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (w - 100, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Status panel
        cv2.rectangle(annotated, (w - 200, 50), (w - 10, 150), (0, 0, 0), -1)
        cv2.putText(annotated, f"Tracked: {self.tracker.get_object_count()}", 
                   (w - 190, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated, f"Known: {len(self.known_faces)}", 
                   (w - 190, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated, f"Present: {len(self.attendance_records)}", 
                   (w - 190, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Log messages
        for i, (msg, ts) in enumerate(self.log_messages[-3:]):
            alpha = 1.0 - (time.time() - ts) / 3.0
            color = (0, int(255 * alpha), 0)
            cv2.putText(annotated, msg, (15, 250 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Controls
        cv2.putText(annotated, "Q:Quit | S:Save | G:Gesture | L:Liveness", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return annotated

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()

            # Preprocess
            processed, _ = self.preprocessor.process(frame)

            # Detect faces
            faces = self.detector.detect_faces(processed)

            # Recognize
            names = []
            bboxes = []
            for face in faces:
                x, y, w, h = face["bbox"]
                bboxes.append((x, y, w, h))
                roi = processed[y:y+h, x:x+w]
                name, conf = self.recognize_face_demo(roi)
                names.append(name)
                face["name"] = name

            # Track
            tracked = self.tracker.update(bboxes, names)

            # Mark attendance
            for oid, centroid, name, is_new, marked in tracked:
                if name != "Unknown" and not marked and not is_new:
                    pid = self.known_faces.get(name, (None, "N/A"))[1]
                    success, msg = self.mark_attendance(name, pid)
                    if success:
                        self.tracker.mark_attendance(oid)
                        self.add_log(f"Marked: {name}")
                        print(f"[DEMO] {msg}")

            # Gesture
            if self.gesture_enabled and len(faces) > 0:
                x, y, w, h = faces[0]["bbox"]
                wave, score, _ = self.gesture.detect(processed, (x, y, w, h))
                if wave:
                    name = faces[0].get("name", "Unknown")
                    if name != "Unknown":
                        self.add_log(f"Wave: {name}")

            # Draw
            display = self.draw_ui(processed, faces, tracked)
            cv2.imshow("Smart Attendance - DEMO", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self.save_csv()
                self.add_log("Report saved!")
            elif key == ord('g'):
                self.gesture_enabled = not self.gesture_enabled
            elif key == ord('l'):
                self.liveness_enabled = not self.liveness_enabled

        self.save_csv()
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DEMO] Session complete. {len(self.attendance_records)} people marked present.")


if __name__ == "__main__":
    DemoAttendanceSystem().run()
