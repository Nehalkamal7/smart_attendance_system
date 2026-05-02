"""
Smart Attendance System - Main Application  (Enhanced UI Edition)
================================================================
Addresses all 8 Rubric Sections with a modern dark-theme UI.

Usage:
    python main.py

Keys:
    Q - Quit
    S - Save report
    R - Register new face
    G - Toggle gesture detection
    L - Toggle liveness verification
    D - Open web dashboard (browser)
"""

import cv2
import numpy as np
import time
import threading
import webbrowser
from datetime import datetime

# ── Project modules ────────────────────────────────────────────────────────────
from config import (
    PROJECT_TITLE, VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT,
    OUTPUT, DISPLAY, INTERACTION, PREPROCESSING
)
from preprocessing import Preprocessor
from detector import FaceDetector
from recognizer import FaceRecognizer
from tracker import CentroidTracker
from gesture import WaveDetector
from database_manager import AttendanceManager
from ui_renderer import UIRenderer


class SmartAttendanceSystem:
    """Main application class for the Smart Attendance System (Enhanced UI)."""

    # ── Initialisation ─────────────────────────────────────────────────────────
    def __init__(self):
        print("=" * 70)
        print(f"  {PROJECT_TITLE}")
        print("=" * 70)

        # Core components
        self.preprocessor = Preprocessor()
        self.detector     = FaceDetector()
        self.recognizer   = FaceRecognizer()
        self.tracker      = CentroidTracker()
        self.gesture      = WaveDetector()
        self.attendance   = AttendanceManager()

        # State flags
        self.gesture_enabled  = INTERACTION["gesture_enabled"]
        self.liveness_enabled = True
        self.running          = True
        self.register_mode    = False
        self.register_name    = ""

        # Performance
        self.fps         = 0.0
        self.frame_count = 0
        self.start_time  = time.time()

        # Notification queue  [(message, timestamp)]
        self.log_messages: list[tuple[str, float]] = []

        # Video init first so we know frame size
        self._init_video()

        # UI renderer (needs actual frame dimensions)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ui = UIRenderer(actual_w, actual_h)

        print("[INFO] System initialised successfully.")
        print("[INFO] Press Q=Quit  S=Save  R=Register  G=Gesture  L=Liveness  D=Dashboard")
        print("-" * 70)

    # ── Video ──────────────────────────────────────────────────────────────────
    def _init_video(self):
        """Initialize the video capture with optimized settings for Windows."""
        import os
        if os.name == 'nt' and isinstance(VIDEO_SOURCE, int):
            self.cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(VIDEO_SOURCE)
            
        if isinstance(VIDEO_SOURCE, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)
            
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {VIDEO_SOURCE}")
        print(f"[INFO] Video source opened: {VIDEO_SOURCE}")

    # ── Notifications ──────────────────────────────────────────────────────────
    def add_log(self, message: str):
        now = time.time()
        self.log_messages.append((message, now))
        cutoff = DISPLAY["log_display_duration"]
        self.log_messages = [(m, t) for m, t in self.log_messages if now - t < cutoff]
        print(f"[LOG] {message}")

    # ── Frame processing ───────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list, list]:
        """Run the full vision pipeline and return (processed_frame, faces, tracked_objects)."""

        # 4. Preprocessing
        processed, _ = self.preprocessor.process(frame)

        # 5. Detection
        faces = self.detector.detect_faces(processed)

        # 6. Recognition + liveness
        face_bboxes      = []
        recognized_names = []

        for face in faces:
            x, y, w, h = face["bbox"]
            face_bboxes.append((x, y, w, h))
            face_roi = processed[y:y + h, x:x + w]

            is_live = True
            if self.liveness_enabled and face_roi.size > 0:
                is_live, _, _ = self.detector.verify_liveness(face_roi)

            name, confidence = "Unknown", 0.0
            if is_live and face_roi.size > 0:
                if self.detector.method == "face_recognition":
                    name, confidence, _ = self.recognizer.recognize_face(
                        processed, (y, x + w, y + h, x)
                    )
                else:
                    name, confidence, _ = self.recognizer.recognize_face(face_roi)

            recognized_names.append(name)
            face["name"]       = name
            face["confidence"] = confidence
            face["is_live"]    = is_live

        # 6b. Tracking
        tracked_objects = self.tracker.update(face_bboxes, recognized_names)

        # Mark attendance
        for oid, centroid, name, is_new, marked in tracked_objects:
            if name != "Unknown" and not marked and not is_new:
                if self.tracker.disappeared.get(oid, 0) == 0:
                    success, msg, is_new_entry = self.attendance.mark_attendance(
                        name, confidence=0.9, method="auto"
                    )
                    if success and is_new_entry:
                        self.tracker.mark_attendance(oid)
                        self.add_log(f"Marked: {name}")

        # 8. Gesture detection
        if self.gesture_enabled and faces:
            x, y, w, h = faces[0]["bbox"]
            wave_detected, _, _ = self.gesture.detect(processed, (x, y, w, h))
            if wave_detected:
                name = faces[0].get("name", "Unknown")
                if name != "Unknown":
                    success, msg, _ = self.attendance.mark_attendance(
                        name, confidence=0.95, method="gesture"
                    )
                    if success:
                        self.add_log(f"Wave confirmed: {name}")

        return processed, faces, tracked_objects

    # ── Key handling ───────────────────────────────────────────────────────────
    def handle_key(self, key: int):
        if key == INTERACTION["key_quit"]:
            self.running = False

        elif key == INTERACTION["key_save"]:
            self.attendance.save_csv()
            self.attendance.save_excel()
            _, path = self.attendance.generate_daily_report()
            self.add_log("Report saved!")

        elif key == INTERACTION["key_register"]:
            self.register_mode = True
            self.register_name = ""

        elif key == INTERACTION["key_toggle_gesture"]:
            self.gesture_enabled = not self.gesture_enabled
            self.add_log(f"Gesture: {'ON' if self.gesture_enabled else 'OFF'}")

        elif key == INTERACTION["key_toggle_liveness"]:
            self.liveness_enabled = not self.liveness_enabled
            self.add_log(f"Liveness: {'ON' if self.liveness_enabled else 'OFF'}")

        elif key == ord('d') or key == ord('D'):
            # Open web dashboard
            try:
                webbrowser.open("http://127.0.0.1:5000")
                self.add_log("Dashboard opened in browser")
            except Exception:
                self.add_log("Could not open dashboard")

        # Register mode – capture typed name
        if self.register_mode:
            if key == 13:  # ENTER
                if self.register_name.strip():
                    self._register_current_face()
                self.register_mode = False
            elif key == 27:  # ESC
                self.register_mode = False
                self.register_name = ""
            elif 32 <= key <= 126:
                self.register_name += chr(key)
            elif key == 8 and self.register_name:  # BACKSPACE
                self.register_name = self.register_name[:-1]

    def _register_current_face(self):
        """Capture the current frame and register the face."""
        ret, frame = self.cap.read()
        if ret:
            success, msg = self.recognizer.register_face(frame, self.register_name.strip())
            self.add_log(msg)
            print(f"[REGISTER] {msg}")

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self):
        print("[INFO] Entering main loop …")

        # Make window resizable & full-screen friendly
        cv2.namedWindow(DISPLAY["window_name"],
                        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(DISPLAY["window_name"], FRAME_WIDTH, FRAME_HEIGHT)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # FPS calculation
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time  = time.time()

            # Vision pipeline
            processed, faces, tracked_objects = self.process_frame(frame)

            # Gather context for UI
            stats      = self.attendance.get_statistics()
            records    = self.attendance.get_today_records()
            reg_users  = self.recognizer.get_registered_users()
            preproc    = self.preprocessor.get_processing_summary()

            context = {
                "fps":             self.fps,
                "total_today":     stats["total_today"],
                "tracking_count":  self.tracker.get_object_count(),
                "gesture_on":      self.gesture_enabled,
                "liveness_on":     self.liveness_enabled,
                "preproc_summary": preproc,
                "stats":           stats,
                "registered_users": reg_users,
                "records":         records,
                "log_messages":    self.log_messages,
                "current_time":    time.time(),
                "log_duration":    DISPLAY["log_display_duration"],
                "faces":           faces,
                "tracked_objects": tracked_objects,
                "register_mode":   self.register_mode,
                "register_name":   self.register_name,
            }

            # Render UI on top of processed frame
            display_frame = self.ui.draw_frame(processed, context)

            cv2.imshow(DISPLAY["window_name"], display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.handle_key(key)

        self.shutdown()

    # ── Shutdown ───────────────────────────────────────────────────────────────
    def shutdown(self):
        print("\n[INFO] Shutting down …")
        self.attendance.save_csv()
        self.attendance.save_excel()
        _, path = self.attendance.generate_daily_report()
        print(f"[INFO] Report saved: {path}")

        self.cap.release()
        cv2.destroyAllWindows()

        stats = self.attendance.get_statistics()
        print("\n" + "=" * 70)
        print("  SESSION SUMMARY")
        print("=" * 70)
        print(f"  Total attendance today : {stats['total_today']}")
        print(f"  Total logged all time  : {stats['total_all_time']}")
        print("=" * 70)
        print("[INFO] Shutdown complete.")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    try:
        system = SmartAttendanceSystem()
        system.run()
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
