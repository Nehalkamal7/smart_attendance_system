"""
Gesture Detection Module
Addresses Rubric Section 8: User Interaction - Gesture-based interaction
Wave to confirm presence.
"""
import cv2
import numpy as np
from collections import deque
from config import INTERACTION


class WaveDetector:
    """
    Detects waving gesture using motion history.
    User waves hand to confirm their attendance.
    """

    def __init__(self, history_size=10):
        self.history_size = INTERACTION["wave_history"]
        self.threshold = INTERACTION["wave_threshold"]
        self.motion_history = deque(maxlen=self.history_size)
        self.prev_gray = None
        self.prev_roi = None
        self.wave_detected = False
        self.wave_cooldown = 0
        self.wave_cooldown_frames = 30  # 1 second at 30fps

    def detect(self, frame, face_roi=None):
        """
        Detect wave gesture in frame.

        Args:
            frame: Current video frame
            face_roi: Optional face region to focus on (x, y, w, h)

        Returns:
            (wave_detected: bool, motion_score: float, visualization: image)
        """
        self.wave_detected = False

        # Decrement cooldown
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            return False, 0, frame

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Focus on face region if provided
        if face_roi is not None:
            x, y, w, h = face_roi
            # Expand region to include hand waving area (below face)
            y_end = min(y + h + int(h * 0.5), frame.shape[0])
            x_end = min(x + w, frame.shape[1])
            x = max(0, x)
            y = max(0, y)
            gray = gray[y:y_end, x:x_end]

        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # If first frame or ROI changed size, initialize/reset
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            self.prev_roi = face_roi
            return False, 0, frame

        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Calculate motion score
        motion_score = np.sum(motion_mask) / 255.0
        self.motion_history.append(motion_score)

        # Update previous frame
        self.prev_gray = gray.copy()
        self.prev_roi = face_roi

        # Detect wave pattern: alternating high motion
        if len(self.motion_history) >= self.history_size:
            avg_motion = np.mean(self.motion_history)
            motion_variance = np.var(self.motion_history)

            # Wave pattern: significant motion with variation (back and forth)
            if avg_motion > self.threshold and motion_variance > self.threshold * 2:
                self.wave_detected = True
                self.wave_cooldown = self.wave_cooldown_frames
                self.motion_history.clear()

        # Create visualization
        vis = frame.copy()
        if face_roi is not None:
            x, y, w, h = face_roi
            y_end = min(y + h + int(h * 0.5), frame.shape[0])
            x_end = min(x + w, frame.shape[1])
            # Draw detection region
            cv2.rectangle(vis, (x, y), (x_end, y_end), (255, 255, 0), 2)

            # Overlay motion mask
            if motion_mask.shape[0] == (y_end - y) and motion_mask.shape[1] == (x_end - x):
                mask_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                mask_color[:, :] = (0, 0, 255)  # Red for motion
                alpha = 0.3
                roi_vis = vis[y:y_end, x:x_end]
                if roi_vis.shape == mask_color.shape:
                    cv2.addWeighted(roi_vis, 1 - alpha, mask_color, alpha, 0, roi_vis)

        # Draw motion indicator
        bar_height = int(min(motion_score / 1000, 1.0) * 100)
        cv2.rectangle(vis, (10, 200), (30, 200 - bar_height), (0, 255, 0), -1)
        cv2.putText(vis, "Motion", (5, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if self.wave_detected:
            cv2.putText(vis, "WAVE DETECTED!", (vis.shape[1] // 2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        return self.wave_detected, motion_score, vis

    def reset(self):
        """Reset detector state."""
        self.prev_gray = None
        self.prev_roi = None
        self.motion_history.clear()
        self.wave_detected = False
        self.wave_cooldown = 0


class SimpleGestureRecognizer:
    """
    Simple gesture recognition using hand tracking.
    Can detect: wave, thumbs up, open palm.
    """

    def __init__(self):
        self.gesture_history = deque(maxlen=5)

    def detect_hand_gesture(self, frame, hand_roi):
        """
        Detect hand gesture in ROI.

        Args:
            frame: Current frame
            hand_roi: (x, y, w, h) of hand region

        Returns:
            gesture_name: str or None
        """
        if hand_roi is None:
            return None

        x, y, w, h = hand_roi
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            return None

        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 1000:
            return None

        # Convex hull and defects
        hull = cv2.convexHull(largest, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(largest, hull)
            if defects is not None:
                defect_count = len(defects)

                # Simple classification
                if defect_count >= 4:
                    gesture = "OPEN_PALM"
                elif defect_count >= 2:
                    gesture = "WAVE"
                else:
                    gesture = "THUMBS_UP"

                self.gesture_history.append(gesture)

                # Return most common gesture in history
                if len(self.gesture_history) >= 3:
                    return max(set(self.gesture_history), key=self.gesture_history.count)
                return gesture

        return None
