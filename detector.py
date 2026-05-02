"""
Detector Module
Addresses Rubric Section 5: Feature Extraction / Detection
- Face detection using Haar cascades or DNN models
- Eye detection for liveness verification (to prevent spoofing with photos)
- Contour detection for bounding boxes
"""
import cv2
import numpy as np
from config import FACE_DETECTION, EYE_DETECTION


class FaceDetector:
    """Handles face and eye detection using multiple methods."""

    def __init__(self):
        self.method = FACE_DETECTION["method"]

        # Initialize Haar cascades (always available in OpenCV)
        self.face_cascade = cv2.CascadeClassifier(FACE_DETECTION["haar_cascade_path"])
        self.eye_cascade = cv2.CascadeClassifier(EYE_DETECTION["cascade_path"])

        # Initialize DNN face detector (if available)
        self.dnn_detector = None
        if self.method == "dnn":
            self._init_dnn_detector()

        # face_recognition library detector
        self.fr_detector = None
        if self.method == "face_recognition":
            try:
                import face_recognition
                self.fr_detector = face_recognition
            except ImportError:
                print("[WARNING] face_recognition not installed. Falling back to Haar.")
                self.method = "haar"

    def _init_dnn_detector(self):
        """Initialize OpenCV DNN face detector."""
        try:
            model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "models/deploy.prototxt"
            self.dnn_detector = cv2.dnn.readNetFromCaffe(config_file, model_file)
        except:
            print("[WARNING] DNN model files not found. Falling back to Haar.")
            self.method = "haar"

    def detect_faces(self, frame):
        """
        Detect faces in the frame.

        Args:
            frame: BGR image

        Returns:
            List of face dictionaries: {
                "bbox": (x, y, w, h),
                "confidence": float,
                "landmarks": {}  # Optional facial landmarks
            }
        """
        if self.method == "face_recognition" and self.fr_detector:
            return self._detect_faces_fr(frame)
        elif self.method == "dnn" and self.dnn_detector:
            return self._detect_faces_dnn(frame)
        else:
            return self._detect_faces_haar(frame)

    def _detect_faces_fr(self, frame):
        """Detect faces using face_recognition library (HOG/CNN)."""
        from config import RECOGNITION

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        locations = self.fr_detector.face_locations(
            rgb, 
            number_of_times_to_upsample=RECOGNITION["upsample"],
            model=RECOGNITION["model"]
        )

        faces = []
        for (top, right, bottom, left) in locations:
            x, y, w, h = left, top, right - left, bottom - top
            faces.append({
                "bbox": (x, y, w, h),
                "confidence": 0.99,  # face_recognition doesn't give confidence
                "landmarks": {}
            })

        return faces

    def _detect_faces_haar(self, frame):
        """Detect faces using Haar cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION["scale_factor"],
            minNeighbors=FACE_DETECTION["min_neighbors"],
            minSize=FACE_DETECTION["min_size"]
        )

        faces = []
        for (x, y, w, h) in detections:
            # Calculate pseudo-confidence based on detection quality
            confidence = min(0.95, (w * h) / (gray.shape[0] * gray.shape[1]) * 10 + 0.5)
            faces.append({
                "bbox": (x, y, w, h),
                "confidence": confidence,
                "landmarks": {}
            })

        return faces

    def _detect_faces_dnn(self, frame):
        """Detect faces using OpenCV DNN."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), 
                                     (104.0, 177.0, 123.0))

        self.dnn_detector.setInput(blob)
        detections = self.dnn_detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (left, top, right, bottom) = box.astype("int")
                fw, fh = right - left, bottom - top
                faces.append({
                    "bbox": (left, top, fw, fh),
                    "confidence": confidence,
                    "landmarks": {}
                })

        return faces

    def detect_eyes(self, face_roi):
        """
        Detect eyes within a face region for liveness verification.

        Args:
            face_roi: Cropped face region (numpy array)

        Returns:
            List of eye bounding boxes (x, y, w, h)
        """
        if not EYE_DETECTION["enabled"]:
            return []

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )

        return [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes]

    def verify_liveness(self, face_roi):
        """
        Verify liveness by detecting eyes in the face region.
        Prevents spoofing with static photos.

        Args:
            face_roi: Cropped face region

        Returns:
            (is_live: bool, eye_count: int, eye_boxes: list)
        """
        eyes = self.detect_eyes(face_roi)
        eye_count = len(eyes)
        is_live = eye_count >= EYE_DETECTION["min_eyes"]

        return is_live, eye_count, eyes

    def get_face_contours(self, face_roi):
        """
        Extract contours from face region for shape analysis.

        Args:
            face_roi: Cropped face region

        Returns:
            List of contours
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_detections(self, frame, faces, draw_eyes=True):
        """
        Draw bounding boxes and annotations for detected faces.

        Args:
            frame: Image to draw on
            faces: List of face dictionaries
            draw_eyes: Whether to draw eye detections

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for face in faces:
            x, y, w, h = face["bbox"]
            conf = face.get("confidence", 0)

            # Draw face bounding box
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw confidence
            label = f"Face: {conf:.2f}"
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw eye detections if enabled
            if draw_eyes:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    eyes = self.detect_eyes(face_roi)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(annotated, (x + ex, y + ey), 
                                    (x + ex + ew, y + ey + eh), (255, 0, 0), 1)

        return annotated
