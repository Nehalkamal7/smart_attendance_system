"""
Preprocessing Module
Addresses Rubric Section 4: Preprocessing Steps
- Resize frames for faster processing
- Convert to grayscale or HSV for better detection
- Apply Gaussian blur to reduce noise
- Use edge detection (Canny/Sobel) for feature clarity
- Noise reduction
"""
import cv2
import numpy as np
from config import PREPROCESSING


class Preprocessor:
    """Handles all image preprocessing operations for the attendance system."""

    def __init__(self):
        self.config = PREPROCESSING
        self.processing_log = []

    def process(self, frame):
        """
        Apply the complete preprocessing pipeline to a frame.

        Args:
            frame: Input BGR image (numpy array)

        Returns:
            processed_frame: Preprocessed image
            debug_info: Dictionary of intermediate steps for visualization
        """
        debug_info = {"original": frame.copy()}
        processed = frame.copy()
        self.processing_log = []

        # Step 1: Resize for faster processing
        if self.config["resize"]["enabled"]:
            processed = self.resize(processed)
            self.processing_log.append("Resized")
            debug_info["resized"] = processed.copy()

        # Step 2: Noise reduction / Smoothing
        if self.config["noise_reduction"]["enabled"]:
            processed = self.reduce_noise(processed)
            self.processing_log.append("Noise reduced")
            debug_info["denoised"] = processed.copy()

        # Step 3: Gaussian blur
        if self.config["gaussian_blur"]["enabled"]:
            processed = self.gaussian_blur(processed)
            self.processing_log.append("Gaussian blur applied")
            debug_info["blurred"] = processed.copy()

        # Step 4: Grayscale conversion (if needed for detection)
        if self.config["grayscale"]["enabled"]:
            gray = self.to_grayscale(processed)
            self.processing_log.append("Converted to grayscale")
            debug_info["grayscale"] = gray.copy()
            if self.config["grayscale"]["convert_back"]:
                # Keep original for display, return gray for processing
                pass
            else:
                processed = gray

        # Step 5: HSV conversion (for skin detection or color-based features)
        if self.config["hsv"]["enabled"]:
            hsv = self.to_hsv(processed)
            self.processing_log.append("Converted to HSV")
            debug_info["hsv"] = hsv.copy()
            if not self.config["hsv"]["convert_back"]:
                processed = hsv

        # Step 6: Edge detection (Canny)
        if self.config["edge_detection"]["enabled"]:
            edges = self.edge_detection(processed)
            self.processing_log.append("Edge detection (Canny)")
            debug_info["edges"] = edges.copy()
            if self.config["edge_detection"]["convert_back"]:
                # Convert edges to 3-channel for overlay
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                debug_info["edges_bgr"] = edges_bgr
                # Optionally overlay edges on original
                # processed = cv2.addWeighted(processed, 0.8, edges_bgr, 0.2, 0)

        return processed, debug_info

    def resize(self, frame):
        """Resize frame to configured dimensions for faster processing."""
        width = self.config["resize"]["width"]
        height = self.config["resize"]["height"]
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def to_grayscale(self, frame):
        """Convert frame to grayscale."""
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def to_hsv(self, frame):
        """Convert frame to HSV color space."""
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return frame

    def gaussian_blur(self, frame):
        """Apply Gaussian blur to reduce high-frequency noise."""
        kernel = self.config["gaussian_blur"]["kernel_size"]
        sigma = self.config["gaussian_blur"]["sigma"]
        return cv2.GaussianBlur(frame, kernel, sigma)

    def reduce_noise(self, frame):
        """Apply noise reduction using configured method."""
        method = self.config["noise_reduction"]["method"]

        if method == "gaussian":
            return cv2.GaussianBlur(frame, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(frame, 5)
        elif method == "bilateral":
            return cv2.bilateralFilter(frame, 9, 75, 75)
        else:
            return frame

    def edge_detection(self, frame):
        """Apply Canny edge detection for feature clarity."""
        gray = self.to_grayscale(frame) if len(frame.shape) == 3 else frame
        t1 = self.config["edge_detection"]["threshold1"]
        t2 = self.config["edge_detection"]["threshold2"]
        return cv2.Canny(gray, t1, t2)

    def sobel_edge_detection(self, frame):
        """Alternative: Apply Sobel edge detection."""
        gray = self.to_grayscale(frame) if len(frame.shape) == 3 else frame
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        return np.uint8(np.clip(sobel_combined, 0, 255))

    def histogram_equalization(self, frame):
        """Improve contrast using histogram equalization."""
        if len(frame.shape) == 3:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(frame)

    def get_processing_summary(self):
        """Return a summary of preprocessing steps applied."""
        return " → ".join(self.processing_log) if self.processing_log else "No preprocessing"
