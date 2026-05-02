"""
Face Registration Utility
Register new users to the attendance database.

Usage:
    python register_face.py --name "John Doe" [--id "EMP001"]

Or interactive mode:
    python register_face.py
"""
import argparse
import cv2
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FACES_DIR, FRAME_WIDTH, FRAME_HEIGHT
from recognizer import FaceRecognizer
from detector import FaceDetector
from preprocessing import Preprocessor


def register_from_camera(name, person_id=None):
    """Register a face using live camera feed."""
    print(f"\n[REGISTER] Preparing to register: {name}")
    print("[REGISTER] Position your face in the camera and press SPACE to capture.")
    print("[REGISTER] Press ESC to cancel.\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    detector = FaceDetector()
    preprocessor = Preprocessor()
    recognizer = FaceRecognizer()

    captured = False
    best_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        processed, _ = preprocessor.process(frame)
        faces = detector.detect_faces(processed)

        display = processed.copy()

        if len(faces) == 1:
            x, y, w, h = faces[0]["bbox"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "Press SPACE to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            best_frame = processed.copy()
        elif len(faces) > 1:
            cv2.putText(display, "Multiple faces detected!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Register Face", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32 and best_frame is not None:  # SPACE
            captured = True
            break
        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured and best_frame is not None:
        success, msg = recognizer.register_face(best_frame, name, person_id)
        print(f"[REGISTER] {msg}")

        if success:
            # Save face image
            face_path = os.path.join(FACES_DIR, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(face_path, best_frame)
            print(f"[REGISTER] Face image saved: {face_path}")

        return success
    else:
        print("[REGISTER] Registration cancelled.")
        return False


def register_from_image(image_path, name, person_id=None):
    """Register a face from an image file."""
    print(f"\n[REGISTER] Loading image: {image_path}")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return False

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return False

    recognizer = FaceRecognizer()
    success, msg = recognizer.register_face(frame, name, person_id)
    print(f"[REGISTER] {msg}")

    if success:
        face_path = os.path.join(FACES_DIR, f"{name.replace(' ', '_')}.jpg")
        cv2.imwrite(face_path, frame)
        print(f"[REGISTER] Face image saved: {face_path}")

    return success


def list_registered():
    """List all registered users."""
    recognizer = FaceRecognizer()
    users = recognizer.get_registered_users()

    print("\n" + "=" * 50)
    print("  REGISTERED USERS")
    print("=" * 50)

    if not users:
        print("  No users registered yet.")
    else:
        print(f"  {'Name':<25} {'ID':<15}")
        print("-" * 50)
        for name, pid in users:
            print(f"  {name:<25} {pid:<15}")

    print("=" * 50)
    print(f"  Total: {len(users)} users\n")


def delete_user(name):
    """Delete a registered user."""
    recognizer = FaceRecognizer()
    if recognizer.delete_user(name):
        print(f"[REGISTER] Deleted user: {name}")

        # Remove face image
        face_path = os.path.join(FACES_DIR, f"{name.replace(' ', '_')}.jpg")
        if os.path.exists(face_path):
            os.remove(face_path)

        return True
    else:
        print(f"[REGISTER] User not found: {name}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Face Registration Utility")
    parser.add_argument("--name", "-n", help="Person's name")
    parser.add_argument("--id", "-i", help="Person's ID (optional)")
    parser.add_argument("--image", "-img", help="Image file path (instead of camera)")
    parser.add_argument("--list", "-l", action="store_true", help="List registered users")
    parser.add_argument("--delete", "-d", help="Delete user by name")

    args = parser.parse_args()

    if args.list:
        list_registered()
    elif args.delete:
        delete_user(args.delete)
    elif args.name:
        if args.image:
            register_from_image(args.image, args.name, args.id)
        else:
            register_from_camera(args.name, args.id)
    else:
        # Interactive mode
        print("\n" + "=" * 50)
        print("  FACE REGISTRATION UTILITY")
        print("=" * 50)
        print("  1. Register from camera")
        print("  2. Register from image file")
        print("  3. List registered users")
        print("  4. Delete user")
        print("  5. Exit")
        print("=" * 50)

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            name = input("Enter name: ").strip()
            pid = input("Enter ID (optional): ").strip() or None
            register_from_camera(name, pid)
        elif choice == "2":
            img_path = input("Enter image path: ").strip()
            name = input("Enter name: ").strip()
            pid = input("Enter ID (optional): ").strip() or None
            register_from_image(img_path, name, pid)
        elif choice == "3":
            list_registered()
        elif choice == "4":
            name = input("Enter name to delete: ").strip()
            delete_user(name)
        else:
            print("Exiting...")


if __name__ == "__main__":
    main()
