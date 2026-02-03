"""
Frame processing module for Water Drawing App.

Handles image processing with static background subtraction.
No dependencies on other application modules.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def process_frame(
    frame: np.ndarray,
    config: dict,
    reference_bg: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a frame: resize, convert to grayscale, apply background subtraction.
    
    Uses static background subtraction with configurable diff modes.
    
    Args:
        frame: Input BGR frame from camera.
        config: Configuration dictionary with processing settings.
        reference_bg: Reference background grayscale image for subtraction.
                     If None, returns empty binary image.
        
    Returns:
        Tuple of (resized_color_image, binary_image, grayscale_image).
    """
    if config["flip_vertical"]:
        frame = cv2.flip(frame, 0)
    if config["flip_horizontal"]:
        frame = cv2.flip(frame, 1)

    # Resize to output dimensions (aspect ratio ignored to match valve array)
    output_size = (config["output_width"], config["output_height"])
    small = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Apply static background subtraction
    if reference_bg is not None:
        # Compute difference based on diff_mode
        diff_mode = config["diff_mode"]
        
        if diff_mode == "lighter":
            # Only detect pixels brighter than reference (hand lighter than bg)
            diff = cv2.subtract(gray, reference_bg)
        elif diff_mode == "darker":
            # Only detect pixels darker than reference (hand darker than bg / shadows)
            diff = cv2.subtract(reference_bg, gray)
        else:  # "both"
            # Detect any difference (original behavior)
            diff = cv2.absdiff(gray, reference_bg)
        
        # Threshold the difference
        _, binary = cv2.threshold(diff, config["difference_threshold"], 255, cv2.THRESH_BINARY)
    else:
        # No reference background - return empty binary
        binary = np.zeros_like(gray)
    
    # Apply morphological operations if enabled
    morph_erode = config["morph_erode"]
    morph_dilate = config["morph_dilate"]
    
    if morph_erode > 0 or morph_dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if morph_erode > 0:
            binary = cv2.erode(binary, kernel, iterations=morph_erode)
        if morph_dilate > 0:
            binary = cv2.dilate(binary, kernel, iterations=morph_dilate)
    
    return small, binary, gray


def capture_reference_background(
    frame: np.ndarray,
    config: dict
) -> np.ndarray:
    """
    Capture a reference background from a frame.
    
    Args:
        frame: Input BGR frame from camera.
        config: Configuration dictionary with flip settings.
        
    Returns:
        Grayscale reference background at output resolution.
    """
    if config["flip_vertical"]:
        frame = cv2.flip(frame, 0)
    if config["flip_horizontal"]:
        frame = cv2.flip(frame, 1)

    # Resize to output dimensions
    output_size = (config["output_width"], config["output_height"])
    small = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
