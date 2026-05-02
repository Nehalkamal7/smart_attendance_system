"""
Tracker Module
Addresses Rubric Section 6: Computer Vision Technique - Tracking
Follow detected faces across frames to avoid duplicate entries.
Uses centroid tracking algorithm.
"""
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
from config import TRACKING


class CentroidTracker:
    """
    Centroid-based object tracker.
    Maintains tracking of faces across frames to prevent duplicate attendance.
    """

    def __init__(self):
        self.next_object_id = 0
        self.objects = OrderedDict()  # object_id -> centroid
        self.disappeared = OrderedDict()  # object_id -> frames disappeared
        self.attendance_marked = OrderedDict()  # object_id -> bool
        self.names = OrderedDict()  # object_id -> recognized name
        self.last_seen = OrderedDict()  # object_id -> frame count

        self.max_disappeared = TRACKING["max_disappeared"]
        self.max_distance = TRACKING["max_distance"]
        self.frame_count = 0

    def register(self, centroid):
        """Register a new object with the next available ID."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.attendance_marked[self.next_object_id] = False
        self.names[self.next_object_id] = "Unknown"
        self.last_seen[self.next_object_id] = self.frame_count
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.attendance_marked[object_id]
        del self.names[object_id]
        del self.last_seen[object_id]

    def update(self, face_bboxes, recognized_names=None):
        """
        Update tracker with new detections.

        Args:
            face_bboxes: List of (x, y, w, h) bounding boxes
            recognized_names: Optional list of names corresponding to bboxes

        Returns:
            List of tracked objects: [(object_id, centroid, name, is_new, attendance_marked)]
        """
        self.frame_count += 1

        if recognized_names is None:
            recognized_names = ["Unknown"] * len(face_bboxes)

        # Calculate centroids for input bounding boxes
        input_centroids = []
        for (x, y, w, h) in face_bboxes:
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids.append((cx, cy))

        # If no objects being tracked, register all new detections
        if len(self.objects) == 0:
            results = []
            for i, centroid in enumerate(input_centroids):
                oid = self.register(centroid)
                self.names[oid] = recognized_names[i]
                results.append((oid, centroid, recognized_names[i], True, False))
            return results

        # If no new detections, mark all existing as disappeared
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return [(oid, self.objects[oid], self.names[oid], False, self.attendance_marked[oid]) 
                    for oid in self.objects.keys()]

        # Compute distance matrix between existing and new centroids
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), np.array(input_centroids))

        # Find closest matches
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        results = []

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Check if distance is within threshold
            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]

            # Update existing object
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            self.last_seen[object_id] = self.frame_count

            # Update name if recognized
            if recognized_names[col] != "Unknown":
                self.names[object_id] = recognized_names[col]

            used_rows.add(row)
            used_cols.add(col)

            results.append((object_id, input_centroids[col], self.names[object_id], 
                          False, self.attendance_marked[object_id]))

        # Handle unmatched existing objects
        unused_rows = set(range(D.shape[0])).difference(used_rows)
        unused_cols = set(range(D.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
            else:
                results.append((object_id, self.objects[object_id], self.names[object_id],
                              False, self.attendance_marked[object_id]))

        # Register new objects for unmatched detections
        for col in unused_cols:
            oid = self.register(input_centroids[col])
            self.names[oid] = recognized_names[col]
            results.append((oid, input_centroids[col], recognized_names[col], True, False))

        return results

    def mark_attendance(self, object_id):
        """Mark attendance for a tracked object."""
        if object_id in self.attendance_marked:
            self.attendance_marked[object_id] = True
            return True
        return False

    def is_attendance_marked(self, object_id):
        """Check if attendance has been marked for this object."""
        return self.attendance_marked.get(object_id, False)

    def get_object_count(self):
        """Return number of currently tracked objects."""
        return len(self.objects)

    def reset(self):
        """Reset all tracking state."""
        self.next_object_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.attendance_marked.clear()
        self.names.clear()
        self.last_seen.clear()
        self.frame_count = 0
