"""
Water Drawing Calibration App

A responsive video capture and binarization tool for the water drawing exhibit.
Displays original and binary images side-by-side with real-time threshold adjustment.

Key design: Display loop is decoupled from Arduino communication.
Display NEVER waits for Arduino - user always sees real-time video.
"""

import json
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

from arduino import ArduinoSender


# ============== CONFIG ==============

def get_config_path() -> str:
    """Get the path to config.json relative to this script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config(path: str) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        path: Path to config.json file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file is invalid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
        print(f"Configuration loaded from {path}")
        return config


def save_config(path: str, config: dict) -> bool:
    """
    Save configuration to JSON file.
    
    Args:
        path: Path to config.json file.
        config: Configuration dictionary to save.
        
    Returns:
        True if save was successful, False otherwise.
    """
    try:
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {path}")
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


# ============== CAPTURE ==============

def init_capture(config: dict) -> Optional[cv2.VideoCapture]:
    """
    Initialize video capture with settings from config.
    
    Supports both camera index (int) and video file path (string).
    
    Args:
        config: Configuration dictionary with video_source, capture_width, capture_height.
        
    Returns:
        cv2.VideoCapture object if successful, None otherwise.
    """
    video_source = config["video_source"]
    
    # Auto-detect: int = camera index, string = file path
    if isinstance(video_source, str) and not video_source.isdigit():
        # It's a file path
        if not os.path.exists(video_source):
            print(f"Error: Video file not found: {video_source}")
            return None
        cap = cv2.VideoCapture(video_source)
    else:
        # It's a camera index
        camera_index = int(video_source)
        cap = cv2.VideoCapture(camera_index)
        
        # Set capture resolution to reduce USB bandwidth and latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["capture_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["capture_height"])
        
        # Disable auto-adjustment to keep consistent brightness for background subtraction
        # Try multiple approaches since cameras vary wildly in what they accept
        
        # Attempt to disable auto-exposure (different cameras use different values)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode on many cameras
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # Manual mode on other cameras
        
        # Set fixed exposure value (negative = shorter exposure, less light)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Adjust this value if too dark/bright
        
        # Disable auto white balance and set fixed value
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)  # Neutral daylight
        
        # Disable auto gain
        cap.set(cv2.CAP_PROP_GAIN, 0)
        
        # Print what the camera actually accepted
        print(f"Camera settings - Auto-exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}, "
              f"Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}, "
              f"Auto-WB: {cap.get(cv2.CAP_PROP_AUTO_WB)}, "
              f"Gain: {cap.get(cv2.CAP_PROP_GAIN)}")
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return None
    
    # Report actual capture resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture initialized: {actual_w}x{actual_h}")
    
    return cap


# ============== PROCESSING ==============

def process_frame(
    frame: np.ndarray,
    config: dict,
    reference_bg: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a frame: resize, convert to grayscale, apply threshold.
    
    Args:
        frame: Input BGR frame from camera.
        config: Configuration dictionary.
        reference_bg: Optional reference background for static background subtraction.
        
    Returns:
        Tuple of (resized_color_image, binary_image, grayscale_image).
    """
    # Apply flips if configured
    if config["flip_vertical"]:
        frame = cv2.flip(frame, 0)
    if config["flip_horizontal"]:
        frame = cv2.flip(frame, 1)
    
    # Resize to output dimensions (aspect ratio ignored to match valve array)
    output_size = (config["output_width"], config["output_height"])
    small = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Apply static background subtraction if enabled and reference exists
    if reference_bg is not None and config.get("use_static_background", False):
        # Compute difference based on diff_mode
        diff_mode = config.get("diff_mode", "both")  # "both", "lighter", "darker"
        
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
        # Fall back to simple threshold on grayscale
        _, binary = cv2.threshold(gray, config["threshold"], 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations if enabled
    morph_erode = config.get("morph_erode", 0)
    morph_dilate = config.get("morph_dilate", 0)
    
    if morph_erode > 0 or morph_dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if morph_erode > 0:
            binary = cv2.erode(binary, kernel, iterations=morph_erode)
        if morph_dilate > 0:
            binary = cv2.dilate(binary, kernel, iterations=morph_dilate)
    
    return small, binary, gray


# ============== DISPLAY ==============

def create_stacked_display(
    original: np.ndarray,
    binary: np.ndarray,
    scale: int
) -> np.ndarray:
    """
    Create a stacked display with original on top, binary below.
    
    Args:
        original: Color image (BGR).
        binary: Binary image (grayscale).
        scale: Upscale factor for visibility.
        
    Returns:
        Combined display frame.
    """
    # Upscale both images
    h, w = original.shape[:2]
    new_size = (w * scale, h * scale)
    
    original_scaled = cv2.resize(original, new_size, interpolation=cv2.INTER_NEAREST)
    
    # Convert binary to BGR for stacking
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary_scaled = cv2.resize(binary_bgr, new_size, interpolation=cv2.INTER_NEAREST)
    
    # Stack vertically
    stacked = cv2.vconcat([original_scaled, binary_scaled])
    
    return stacked


def draw_overlay(
    frame: np.ndarray, 
    config: dict, 
    fps: float, 
    white_ratio: float,
    has_reference_bg: bool = False
) -> np.ndarray:
    """
    Draw overlay text showing current settings.
    
    Args:
        frame: Frame to draw on.
        config: Configuration dictionary.
        fps: Current frames per second.
        white_ratio: Current white pixel ratio (0-1).
        has_reference_bg: Whether a reference background has been captured.
        
    Returns:
        Frame with overlay.
    """
    # Create a copy to draw on
    display = frame.copy()
    
    # Build settings to display based on mode
    use_static = config.get("use_static_background", False)
    
    if use_static:
        diff_mode = config.get("diff_mode", "both")
        morph_erode = config.get("morph_erode", 0)
        morph_dilate = config.get("morph_dilate", 0)
        lines = [
            f"Diff Threshold: {config['difference_threshold']} (+/- to adjust)",
            f"Diff Mode: {diff_mode} (d=cycle)",
            f"Morph: erode={morph_erode} dilate={morph_dilate} (e/E, l/L)",
            f"FPS: {fps:.1f}  White: {white_ratio * 100:.1f}%",
            f"Static BG: ON (b)  Ref: {'SET' if has_reference_bg else 'NOT SET'} (r)",
            f"Fullscreen: {'ON' if config['fullscreen'] else 'OFF'} (f)",
            "s=save, SPACE=send, q=quit"
        ]
    else:
        lines = [
            f"Threshold: {config['threshold']} (+/- to adjust)",
            f"FPS: {fps:.1f}",
            f"White: {white_ratio * 100:.1f}%",
            f"Static BG: {'ON' if use_static else 'OFF'} (b)",
            f"Fullscreen: {'ON' if config['fullscreen'] else 'OFF'} (f)",
            "s=save, r=capture bg, SPACE=send, q=quit"
        ]
    
    # Draw text
    y = 25
    for line in lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 20
    
    return display


def set_fullscreen(window_name: str, enabled: bool) -> None:
    """
    Toggle fullscreen mode for a window.
    
    Args:
        window_name: Name of the OpenCV window.
        enabled: True to enable fullscreen, False for windowed.
    """
    if enabled:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


# ============== MAIN ==============

def main():
    """Main entry point for the calibration app."""
    import time
    
    # Load configuration
    config_path = get_config_path()
    config = load_config(config_path)
    
    # Initialize video capture
    cap = init_capture(config)
    if cap is None:
        print("Failed to initialize video capture. Exiting.")
        sys.exit(1)
    
    # Initialize Arduino sender (mock mode)
    arduino = ArduinoSender(config, mock=True)
    
    # Create window
    window_name = "Water Drawing Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set initial fullscreen state
    if config["fullscreen"]:
        set_fullscreen(window_name, True)
    
    # State variables
    prev_white_ratio = 0.0
    frame_count = 0
    fps = 0.0
    fps_update_time = cv2.getTickCount()
    
    # Static background state
    reference_bg: Optional[np.ndarray] = None
    
    print("\n=== Water Drawing Calibration App ===")
    print("Controls:")
    print("  +/- or LEFT/RIGHT: Adjust threshold")
    print("  d: Cycle diff mode (both/lighter/darker)")
    print("  e/E: Increase/decrease erode iterations")
    print("  l/L: Increase/decrease dilate iterations")
    print("  s: Save config")
    print("  b: Toggle static background mode")
    print("  r: Capture reference background (clear scene first!)")
    print("  f: Toggle fullscreen")
    print("  SPACE: Manual send to Arduino")
    print("  q/ESC: Quit")
    print("=====================================\n")
    
    # Capture initial reference background after a short delay
    print("Capturing initial reference background in 2 seconds...")
    print("Please ensure the scene is EMPTY (no hands)!")
    time.sleep(2)
    
    # Read a few frames to stabilize camera
    for _ in range(10):
        cap.read()
    
    ret, init_frame = cap.read()
    if ret:
        # Process to get grayscale at output resolution
        if config["flip_vertical"]:
            init_frame = cv2.flip(init_frame, 0)
        if config["flip_horizontal"]:
            init_frame = cv2.flip(init_frame, 1)
        output_size = (config["output_width"], config["output_height"])
        small_init = cv2.resize(init_frame, output_size, interpolation=cv2.INTER_AREA)
        reference_bg = cv2.cvtColor(small_init, cv2.COLOR_BGR2GRAY)
        print("Initial reference background captured!")
    else:
        print("Warning: Could not capture initial reference background")
    
    running = True
    while running:
        # -------- CAPTURE (always runs at camera speed) --------
        ret, frame = cap.read()
        if not ret:
            # For video files, loop back to start
            if isinstance(config["video_source"], str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("Error: Could not read frame")
            break
        
        # -------- PROCESS --------
        small, binary, gray = process_frame(frame, config, reference_bg)
        
        # -------- DISPLAY (always updates - never blocked by Arduino) --------
        display_frame = create_stacked_display(small, binary, config["display_scale"])
        
        # Calculate white ratio for change detection
        total_pixels = config["output_width"] * config["output_height"] * 255
        white_ratio = np.sum(binary) / total_pixels
        
        # Draw overlay
        display_frame = draw_overlay(
            display_frame, config, fps, white_ratio,
            has_reference_bg=(reference_bg is not None)
        )
        
        # Show frame
        cv2.imshow(window_name, display_frame)
        
        # -------- CHANGE DETECTION (non-blocking) --------
        change = abs(white_ratio - prev_white_ratio)
        object_appeared = change > config["change_detection_threshold"]
        
        if object_appeared and arduino.is_ready():
            arduino.send_frame(binary)
        
        prev_white_ratio = white_ratio
        
        # -------- FPS CALCULATION --------
        frame_count += 1
        current_time = cv2.getTickCount()
        elapsed = (current_time - fps_update_time) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_update_time = current_time
        
        # -------- HANDLE KEYS --------
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            running = False
            
        elif key in [ord('+'), ord('='), 83, 0]:  # + or right arrow
            # Adjust threshold based on mode
            if config.get("use_static_background", False):
                step = config.get("difference_threshold_step", 5)
                config["difference_threshold"] = min(255, config["difference_threshold"] + step)
                print(f"Difference threshold: {config['difference_threshold']}")
            else:
                config["threshold"] = min(255, config["threshold"] + config["threshold_step"])
                print(f"Threshold: {config['threshold']}")
            
        elif key in [ord('-'), 81, 1]:  # - or left arrow
            # Adjust threshold based on mode
            if config.get("use_static_background", False):
                step = config.get("difference_threshold_step", 5)
                config["difference_threshold"] = max(0, config["difference_threshold"] - step)
                print(f"Difference threshold: {config['difference_threshold']}")
            else:
                config["threshold"] = max(0, config["threshold"] - config["threshold_step"])
                print(f"Threshold: {config['threshold']}")
            
        elif key == ord('s'):
            save_config(config_path, config)
            
        elif key == ord('b'):
            config["use_static_background"] = not config.get("use_static_background", False)
            mode = "Static BG" if config["use_static_background"] else "Simple threshold"
            print(f"Mode: {mode}")
            
        elif key == ord('r'):
            # Capture new reference background
            reference_bg = gray.copy()
            print("Reference background captured!")
            
        elif key == ord('f'):
            config["fullscreen"] = not config["fullscreen"]
            set_fullscreen(window_name, config["fullscreen"])
            print(f"Fullscreen: {'ON' if config['fullscreen'] else 'OFF'}")
        
        elif key == ord('d'):
            # Cycle diff mode: both -> lighter -> darker -> both
            modes = ["both", "lighter", "darker"]
            current = config.get("diff_mode", "both")
            idx = (modes.index(current) + 1) % len(modes)
            config["diff_mode"] = modes[idx]
            print(f"Diff mode: {config['diff_mode']}")
        
        elif key == ord('e'):
            # Increase erode iterations
            config["morph_erode"] = config.get("morph_erode", 0) + 1
            print(f"Morph erode: {config['morph_erode']}")
        
        elif key == ord('E'):
            # Decrease erode iterations
            config["morph_erode"] = max(0, config.get("morph_erode", 0) - 1)
            print(f"Morph erode: {config['morph_erode']}")
        
        elif key == ord('l'):
            # Increase dilate iterations
            config["morph_dilate"] = config.get("morph_dilate", 0) + 1
            print(f"Morph dilate: {config['morph_dilate']}")
        
        elif key == ord('L'):
            # Decrease dilate iterations
            config["morph_dilate"] = max(0, config.get("morph_dilate", 0) - 1)
            print(f"Morph dilate: {config['morph_dilate']}")
            
        elif key == ord(' '):  # SPACE - manual send
            print("Manual send triggered")
            arduino.send_frame(binary)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
