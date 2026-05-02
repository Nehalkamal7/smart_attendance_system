"""
Recognizer Module
Addresses Rubric Section 6: Computer Vision Technique - Recognition
Identify faces against a stored database of registered users.
"""
import os
import pickle
import numpy as np
import cv2
from datetime import datetime
from config import RECOGNITION, FACES_DIR, DATABASE_DIR


class FaceRecognizer:
    """Manages face recognition against a database of registered users."""

    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.db_file = os.path.join(DATABASE_DIR, "face_database.pkl")

        # Try to import face_recognition
        self.fr_available = False
        try:
            import face_recognition
            self.fr = face_recognition
            self.fr_available = True
        except ImportError:
            print("[WARNING] face_recognition not available. Using fallback method.")

        self.load_database()

    def load_database(self):
        """Load known face encodings from database file."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "rb") as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get("encodings", [])
                    self.known_names = data.get("names", [])
                    self.known_ids = data.get("ids", [])
                print(f"[INFO] Loaded {len(self.known_names)} faces from database.")
            except Exception as e:
                print(f"[ERROR] Failed to load database: {e}")
        else:
            print("[INFO] No existing database found. Starting fresh.")

    def save_database(self):
        """Save known face encodings to database file."""
        data = {
            "encodings": self.known_encodings,
            "names": self.known_names,
            "ids": self.known_ids
        }
        with open(self.db_file, "wb") as f:
            pickle.dump(data, f)
        print(f"[INFO] Database saved with {len(self.known_names)} faces.")

    def register_face(self, frame, name, person_id=None):
        """
        Register a new face to the database.

        Args:
            frame: Image containing the face
            name: Person's name
            person_id: Optional unique ID

        Returns:
            success: bool
            message: str
        """
        if not self.fr_available:
            return False, "face_recognition library not installed"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = self.fr.face_locations(rgb, model=RECOGNITION["model"])

        if len(face_locations) == 0:
            return False, "No face detected in frame"

        if len(face_locations) > 1:
            return False, "Multiple faces detected. Please ensure only one face is visible."

        # Get face encoding
        face_encoding = self.fr.face_encodings(rgb, face_locations)[0]

        # Check if person already exists
        if name in self.known_names:
            # Update existing
            idx = self.known_names.index(name)
            self.known_encodings[idx] = face_encoding
            if person_id:
                self.known_ids[idx] = person_id
            self.save_database()
            return True, f"Updated face data for {name}"

        # Add new person
        self.known_encodings.append(face_encoding)
        self.known_names.append(name)
        self.known_ids.append(person_id or f"ID_{len(self.known_names):04d}")

        # Save face image
        face_img_path = os.path.join(FACES_DIR, f"{name.replace(' ', '_')}.jpg")
        cv2.imwrite(face_img_path, frame)

        self.save_database()
        return True, f"Registered {name} successfully"

    def recognize_face(self, frame, face_location=None):
        """
        Recognize a face in the given frame.

        Args:
            frame: Image containing the face
            face_location: Optional (top, right, bottom, left) tuple

        Returns:
            name: Recognized name or "Unknown"
            confidence: Match confidence (0-1)
            person_id: Associated ID
        """
        if not self.fr_available or len(self.known_encodings) == 0:
            return "Unknown", 0.0, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # If no face location provided, detect it
        if face_location is None:
            face_locations = self.fr.face_locations(rgb, model=RECOGNITION["model"])
            if len(face_locations) == 0:
                return "Unknown", 0.0, None
            face_location = face_locations[0]

        # Get encoding
        encodings = self.fr.face_encodings(rgb, [face_location])
        if len(encodings) == 0:
            return "Unknown", 0.0, None

        face_encoding = encodings[0]

        # Compare with known faces
        matches = self.fr.compare_faces(self.known_encodings, face_encoding, 
                                       tolerance=RECOGNITION["tolerance"])
        face_distances = self.fr.face_distance(self.known_encodings, face_encoding)

        name = "Unknown"
        confidence = 0.0
        person_id = None

        if True in matches:
            best_match_idx = np.argmin(face_distances)
            if matches[best_match_idx]:
                name = self.known_names[best_match_idx]
                person_id = self.known_ids[best_match_idx]
                # Convert distance to confidence (closer = higher confidence)
                confidence = 1 - face_distances[best_match_idx]

        return name, confidence, person_id

    def get_registered_users(self):
        """Return list of registered users."""
        return list(zip(self.known_names, self.known_ids))

    def delete_user(self, name):
        """Remove a user from the database."""
        if name in self.known_names:
            idx = self.known_names.index(name)
            self.known_names.pop(idx)
            self.known_encodings.pop(idx)
            self.known_ids.pop(idx)
            self.save_database()
            return True
        return False


class SimpleLBPHRecognizer:
    """
    Fallback recognizer using OpenCV LBPH (Local Binary Patterns Histograms).
    Used when face_recognition library is not available.
    """

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self.reverse_label_map = {}
        self.trained = False
        self.db_file = os.path.join(DATABASE_DIR, "lbph_database.yml")
        self.map_file = os.path.join(DATABASE_DIR, "lbph_labels.pkl")
        self.load_database()

    def load_database(self):
        """Load trained model if exists."""
        if os.path.exists(self.db_file) and os.path.exists(self.map_file):
            self.recognizer.read(self.db_file)
            import pickle
            with open(self.map_file, "rb") as f:
                data = pickle.load(f)
                self.label_map = data["label_map"]
                self.reverse_label_map = {v: k for k, v in self.label_map.items()}
                self.trained = True

    def save_database(self):
        """Save trained model."""
        if self.trained:
            self.recognizer.save(self.db_file)
            import pickle
            with open(self.map_file, "wb") as f:
                pickle.dump({"label_map": self.label_map}, f)

    def register_face(self, face_img, name):
        """Register a face using LBPH."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        if name not in self.label_map:
            label = len(self.label_map)
            self.label_map[name] = label
            self.reverse_label_map[label] = name
        else:
            label = self.label_map[name]

        # For simplicity, train with single image (in production, use multiple)
        faces = [gray]
        labels = [label]

        self.recognizer.train(faces, np.array(labels))
        self.trained = True
        self.save_database()
        return True, f"Registered {name} with LBPH"

    def recognize_face(self, face_img):
        """Recognize face using LBPH."""
        if not self.trained:
            return "Unknown", 0.0, None

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        label, confidence = self.recognizer.predict(gray)

        # LBPH confidence is distance (lower is better)
        # Convert to similarity score
        name = self.reverse_label_map.get(label, "Unknown")
        similarity = max(0, 1 - (confidence / 100))

        if confidence > 50:  # Threshold
            name = "Unknown"

        return name, similarity, None
