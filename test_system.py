"""
Test Suite for Smart Attendance System
Validates all components independently.
"""
import os
import sys
import cv2
import numpy as np
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import Preprocessor
from detector import FaceDetector
from tracker import CentroidTracker
from database_manager import AttendanceManager


class TestPreprocessor(unittest.TestCase):
    """Test preprocessing pipeline."""

    def setUp(self):
        self.preprocessor = Preprocessor()
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_resize(self):
        result = self.preprocessor.resize(self.test_frame)
        self.assertEqual(result.shape[1], 640)
        self.assertEqual(result.shape[0], 480)

    def test_grayscale(self):
        result = self.preprocessor.to_grayscale(self.test_frame)
        self.assertEqual(len(result.shape), 2)

    def test_gaussian_blur(self):
        result = self.preprocessor.gaussian_blur(self.test_frame)
        self.assertEqual(result.shape, self.test_frame.shape)

    def test_edge_detection(self):
        result = self.preprocessor.edge_detection(self.test_frame)
        self.assertEqual(len(result.shape), 2)

    def test_full_pipeline(self):
        result, debug = self.preprocessor.process(self.test_frame)
        self.assertIsNotNone(result)
        self.assertIn("original", debug)


class TestDetector(unittest.TestCase):
    """Test face and eye detection."""

    def setUp(self):
        self.detector = FaceDetector()
        # Create a synthetic face-like image
        self.test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Draw a face-like rectangle
        cv2.rectangle(self.test_frame, (250, 150), (390, 330), (200, 150, 100), -1)
        cv2.circle(self.test_frame, (300, 220), 15, (255, 255, 255), -1)  # eye
        cv2.circle(self.test_frame, (340, 220), 15, (255, 255, 255), -1)  # eye

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.face_cascade)
        self.assertIsNotNone(self.detector.eye_cascade)

    def test_detect_faces(self):
        faces = self.detector.detect_faces(self.test_frame)
        self.assertIsInstance(faces, list)
        # Note: Haar may or may not detect synthetic faces

    def test_detect_eyes(self):
        roi = self.test_frame[150:330, 250:390]
        eyes = self.detector.detect_eyes(roi)
        self.assertIsInstance(eyes, list)


class TestTracker(unittest.TestCase):
    """Test centroid tracking."""

    def setUp(self):
        self.tracker = CentroidTracker()

    def test_register(self):
        oid = self.tracker.register((100, 100))
        self.assertEqual(oid, 0)
        self.assertIn(0, self.tracker.objects)

    def test_update(self):
        # First frame - two faces
        bboxes = [(100, 100, 50, 50), (300, 200, 50, 50)]
        names = ["Person_A", "Person_B"]
        result = self.tracker.update(bboxes, names)
        self.assertEqual(len(result), 2)

        # Second frame - same faces moved slightly
        bboxes = [(105, 105, 50, 50), (305, 205, 50, 50)]
        names = ["Person_A", "Person_B"]
        result = self.tracker.update(bboxes, names)
        self.assertEqual(len(result), 2)

        # Check IDs persisted
        ids = [r[0] for r in result]
        self.assertEqual(sorted(ids), [0, 1])

    def test_deregister(self):
        oid = self.tracker.register((100, 100))
        self.tracker.deregister(oid)
        self.assertNotIn(oid, self.tracker.objects)

    def test_attendance_marking(self):
        oid = self.tracker.register((100, 100))
        self.assertFalse(self.tracker.is_attendance_marked(oid))
        self.tracker.mark_attendance(oid)
        self.assertTrue(self.tracker.is_attendance_marked(oid))


class TestAttendanceManager(unittest.TestCase):
    """Test attendance logging."""

    def setUp(self):
        self.manager = AttendanceManager()
        self.manager.records.clear()
        self.manager.attendance_cooldown.clear()

    def test_mark_attendance(self):
        success, msg, is_new = self.manager.mark_attendance("TestUser", "ID001", 0.95)
        self.assertTrue(success)
        self.assertTrue(is_new)
        self.assertIn("TestUser", self.manager.records)

    def test_cooldown(self):
        self.manager.mark_attendance("TestUser", "ID001", 0.95)
        success, msg, is_new = self.manager.mark_attendance("TestUser", "ID001", 0.95)
        self.assertFalse(success)
        self.assertIn("Cooldown", msg)

    def test_get_today_records(self):
        self.manager.mark_attendance("User1", "ID001", 0.9)
        self.manager.mark_attendance("User2", "ID002", 0.8)
        records = self.manager.get_today_records()
        self.assertEqual(len(records), 2)

    def test_statistics(self):
        self.manager.mark_attendance("User1", "ID001", 0.9)
        stats = self.manager.get_statistics()
        self.assertEqual(stats["total_today"], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def test_pipeline_components(self):
        """Verify all components can be instantiated."""
        preprocessor = Preprocessor()
        detector = FaceDetector()
        tracker = CentroidTracker()
        attendance = AttendanceManager()

        self.assertIsNotNone(preprocessor)
        self.assertIsNotNone(detector)
        self.assertIsNotNone(tracker)
        self.assertIsNotNone(attendance)

    def test_frame_processing(self):
        """Test processing a frame through the pipeline."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        preprocessor = Preprocessor()
        processed, debug = preprocessor.process(frame)

        detector = FaceDetector()
        faces = detector.detect_faces(processed)

        self.assertIsInstance(faces, list)
        self.assertIsInstance(processed, np.ndarray)


def run_tests():
    """Run all tests and display results."""
    print("=" * 60)
    print("  SMART ATTENDANCE SYSTEM - TEST SUITE")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestAttendanceManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success: {'YES' if result.wasSuccessful() else 'NO'}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
