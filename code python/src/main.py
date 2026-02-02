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

def set_camera_controls_linux(device_index: int, config: dict) -> None:
    """
    Set camera controls on Linux using v4l2-ctl.
    
    Args:
        device_index: Camera index (0, 1, 2, etc.)
        config: Configuration dictionary with camera settings.
    """
    import subprocess
    import time
    
    device = f"/dev/video{device_index}"
    
    # Get configured values or use defaults
    exposure = config.get("camera_exposure", 157)
    wb_temp = config.get("camera_wb_temperature", 4600)
    gain = config.get("camera_gain", 0)
    
    # First, disable all auto modes (must happen before setting manual values)
    auto_disable_commands = [
        # Disable auto-exposure (1 = manual mode for most cameras)
        f"v4l2-ctl -d {device} --set-ctrl=exposure_auto=1",
        # Disable auto white balance
        f"v4l2-ctl -d {device} --set-ctrl=white_balance_temperature_auto=0",
        # Some cameras use different control names
        f"v4l2-ctl -d {device} --set-ctrl=white_balance_automatic=0",
        f"v4l2-ctl -d {device} --set-ctrl=auto_white_balance=0",
    ]
    
    for cmd in auto_disable_commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=2, text=True)
            if result.returncode == 0:
                print(f"  OK: {cmd.split('--set-ctrl=')[1]}")
        except Exception:
            pass  # Silently ignore - camera may not support this control
    
    # Small delay to let camera apply auto-disable settings
    time.sleep(0.1)
    
    # Now set the manual values
    manual_commands = [
        (f"v4l2-ctl -d {device} --set-ctrl=exposure_absolute={exposure}", "exposure_absolute"),
        (f"v4l2-ctl -d {device} --set-ctrl=white_balance_temperature={wb_temp}", "white_balance_temperature"),
        (f"v4l2-ctl -d {device} --set-ctrl=gain={gain}", "gain"),
    ]
    
    for cmd, name in manual_commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=2, text=True)
            if result.returncode == 0:
                print(f"  OK: {name}")
            else:
                print(f"  FAIL: {name} - {result.stderr.strip()}")
        except Exception as e:
            print(f"  ERROR: {name} - {e}")
    
    # Verify settings were applied
    print(f"\nVerifying camera settings on {device}:")
    verify_cmd = f"v4l2-ctl -d {device} --get-ctrl=exposure_auto,exposure_absolute,white_balance_temperature_auto,white_balance_temperature,gain"
    try:
        result = subprocess.run(verify_cmd, shell=True, capture_output=True, timeout=2, text=True)
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
    except Exception as e:
        print(f"  Could not verify: {e}")
    
    print(f"\nTarget values: exposure={exposure}, wb={wb_temp}, gain={gain}")


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
        
        # On Linux, use V4L2 backend explicitly to avoid GStreamer warnings
        if sys.platform.startswith('linux'):
            print(f"Opening camera {camera_index} with V4L2 backend...")
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return None
        
        # Set capture resolution to reduce USB bandwidth and latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["capture_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["capture_height"])
        
        # On Linux, use v4l2-ctl which works much more reliably than OpenCV properties
        if sys.platform.startswith('linux'):
            set_camera_controls_linux(camera_index, config)
        else:
            # On Windows/Mac, try OpenCV properties (less reliable but worth trying)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode on many cameras
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # Manual mode on other cameras
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)
            cap.set(cv2.CAP_PROP_GAIN, 0)
            
            print(f"Camera settings (OpenCV) - Auto-exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}, "
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
    
    The original (high-res color) image is scaled to match the width of the 
    binary display while preserving aspect ratio. The binary image is scaled
    with nearest-neighbor to show the actual pixel grid.
    
    Args:
        original: Color image (BGR) at capture resolution.
        binary: Binary image (grayscale) at output resolution (e.g., 64x64).
        scale: Upscale factor for the binary image.
        
    Returns:
        Combined display frame.
    """
    # Scale binary with nearest-neighbor to show pixel grid
    bin_h, bin_w = binary.shape[:2]
    binary_display_size = (bin_w * scale, bin_h * scale)
    
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary_scaled = cv2.resize(binary_bgr, binary_display_size, interpolation=cv2.INTER_NEAREST)
    
    # Scale original to match binary display width, preserving aspect ratio
    orig_h, orig_w = original.shape[:2]
    target_width = binary_display_size[0]
    aspect_ratio = orig_h / orig_w
    target_height = int(target_width * aspect_ratio)
    
    original_scaled = cv2.resize(original, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Stack vertically
    stacked = cv2.vconcat([original_scaled, binary_scaled])
    
    return stacked


def draw_overlay(
    frame: np.ndarray, 
    config: dict, 
    fps: float, 
    white_ratio: float,
    has_reference_bg: bool = False,
    stationary_status: str = ""
) -> np.ndarray:
    """
    Draw overlay text showing current settings.
    
    Args:
        frame: Frame to draw on.
        config: Configuration dictionary.
        fps: Current frames per second.
        white_ratio: Current white pixel ratio (0-1).
        has_reference_bg: Whether a reference background has been captured.
        stationary_status: Current stationary detection status string.
        
    Returns:
        Frame with overlay.
    """
    # Create a copy to draw on
    display = frame.copy()
    
    # Build settings to display based on mode
    use_static = config.get("use_static_background", False)
    stationary_delay = config.get("stationary_delay_ms", 500)
    
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
            f"Stationary delay: {stationary_delay}ms  {stationary_status}",
            f"Fullscreen: {'ON' if config['fullscreen'] else 'OFF'} (f)",
            "s=save, SPACE=send, q=quit"
        ]
    else:
        lines = [
            f"Threshold: {config['threshold']} (+/- to adjust)",
            f"FPS: {fps:.1f}",
            f"White: {white_ratio * 100:.1f}%",
            f"Stationary delay: {stationary_delay}ms  {stationary_status}",
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
    frame_count = 0
    fps = 0.0
    fps_update_time = cv2.getTickCount()
    
    # Static background state
    reference_bg: Optional[np.ndarray] = None
    
    # Streaming state
    waiting_for_stationary = False  # Waiting for user to hold still before streaming starts
    stationary_start_time: Optional[float] = None
    is_streaming = False  # Currently streaming frames to Arduino
    last_send_time: float = 0.0  # For rate limiting Arduino sends
    prev_binary: Optional[np.ndarray] = None  # Previous frame for motion detection
    
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
        # Apply flips to full-res frame for high-res preview
        display_original = frame.copy()
        if config["flip_vertical"]:
            display_original = cv2.flip(display_original, 0)
        if config["flip_horizontal"]:
            display_original = cv2.flip(display_original, 1)
        
        display_frame = create_stacked_display(display_original, binary, config["display_scale"])
        
        # Calculate white ratio for presence detection
        total_pixels = config["output_width"] * config["output_height"]
        white_ratio = np.sum(binary) / (total_pixels * 255)
        
        # Calculate pixel movement using XOR comparison (detects position changes, not just amount)
        stationary_threshold = config.get("stationary_threshold", 0.02)
        if prev_binary is not None:
            # XOR: pixels that are different between frames
            pixel_diff = cv2.bitwise_xor(binary, prev_binary)
            # Count changed pixels as ratio of total pixels
            changed_pixels = np.sum(pixel_diff) / 255  # Each changed pixel is 255
            pixel_change_ratio = changed_pixels / total_pixels
            is_currently_still = pixel_change_ratio < stationary_threshold
        else:
            pixel_change_ratio = 0.0
            is_currently_still = False  # First frame, not stationary yet
        
        # Build status string for overlay (shows pixel change % to help tune threshold)
        require_stationary = config.get("require_stationary_for_send", True)
        arduino_send_fps = config.get("arduino_send_fps", 10)
        send_interval = 1.0 / arduino_send_fps if arduino_send_fps > 0 else 0
        
        if is_streaming:
            if require_stationary:
                # Show stationary status and FPS cooldown during streaming
                time_since_send = time.time() - last_send_time
                fps_ready = time_since_send >= send_interval
                fps_status = "ready" if fps_ready else f"wait {(send_interval - time_since_send):.1f}s"
                
                if stationary_start_time is not None:
                    elapsed_ms = (time.time() - stationary_start_time) * 1000
                    stationary_status = f"[HOLD: {elapsed_ms:.0f}ms | FPS: {fps_status}] move:{pixel_change_ratio:.1%}"
                else:
                    stationary_status = f"[MOVING | FPS: {fps_status}] move:{pixel_change_ratio:.1%}"
            else:
                stationary_status = f"[STREAMING @ {arduino_send_fps} FPS] move:{pixel_change_ratio:.1%}"
        elif waiting_for_stationary:
            if stationary_start_time is not None:
                elapsed_ms = (time.time() - stationary_start_time) * 1000
                stationary_status = f"[HOLD STILL: {elapsed_ms:.0f}ms] move:{pixel_change_ratio:.1%}"
            else:
                stationary_status = f"[MOVING {pixel_change_ratio:.1%} > {stationary_threshold:.1%}]"
        else:
            stationary_status = ""
        
        # Draw overlay
        display_frame = draw_overlay(
            display_frame, config, fps, white_ratio,
            has_reference_bg=(reference_bg is not None),
            stationary_status=stationary_status
        )
        
        # Show frame
        cv2.imshow(window_name, display_frame)
        
        # -------- PRESENCE DETECTION & STREAMING (non-blocking) --------
        # Get config values (change and stationary_threshold already calculated above for display)
        stationary_delay_ms = config.get("stationary_delay_ms", 500)
        min_presence = config.get("min_presence_threshold", 0.05)
        arduino_send_fps = config.get("arduino_send_fps", 10)
        send_interval = 1.0 / arduino_send_fps if arduino_send_fps > 0 else 0
        
        # Determine current state
        has_presence = white_ratio > min_presence  # Is there enough white pixels?
        is_stationary = is_currently_still  # Are pixels stable frame-to-frame? (calculated above)
        current_time_sec = time.time()
        
        require_stationary = config.get("require_stationary_for_send", True)
        
        if is_streaming:
            # Currently streaming
            if not has_presence:
                # Presence ended - stop streaming
                is_streaming = False
                stationary_start_time = None
                print("Presence ended, stopping stream")
            elif require_stationary:
                # Require stationary for each send + respect FPS limit
                time_since_last_send = current_time_sec - last_send_time
                fps_ready = time_since_last_send >= send_interval
                
                if not is_stationary:
                    # User is moving - reset stationary timer
                    if stationary_start_time is not None:
                        print(f"Movement during stream (pixel change={pixel_change_ratio:.1%}), waiting for stationary...")
                    stationary_start_time = None
                else:
                    # User is stationary - track time
                    if stationary_start_time is None:
                        stationary_start_time = current_time_sec
                    
                    # Check if both conditions are met: stationary long enough AND FPS interval passed
                    elapsed_stationary_ms = (current_time_sec - stationary_start_time) * 1000
                    stationary_ready = elapsed_stationary_ms >= stationary_delay_ms
                    
                    if stationary_ready and fps_ready and arduino.is_ready():
                        # Both conditions met - send frame
                        arduino.send_frame(binary)
                        last_send_time = current_time_sec
                        stationary_start_time = None  # Reset for next send
                        print(f"Sent frame (stationary for {elapsed_stationary_ms:.0f}ms)")
            else:
                # Just send at configured FPS rate (no stationary requirement)
                if arduino.is_ready() and (current_time_sec - last_send_time) >= send_interval:
                    arduino.send_frame(binary)
                    last_send_time = current_time_sec
        
        elif waiting_for_stationary:
            # Waiting for user to hold still before starting stream
            if not has_presence:
                # User left before becoming stationary
                waiting_for_stationary = False
                stationary_start_time = None
                print("Presence ended before stationary")
            elif not is_stationary:
                # User is still moving - reset the stationary timer
                if stationary_start_time is not None:
                    print(f"Movement detected (pixel change={pixel_change_ratio:.1%}), resetting timer...")
                stationary_start_time = None
            else:
                # User is holding still (pixel change < threshold)
                if stationary_start_time is None:
                    # Just became stationary - start the timer
                    stationary_start_time = current_time_sec
                    print(f"User stationary (pixel change={pixel_change_ratio:.1%}), starting {stationary_delay_ms}ms timer...")
                else:
                    # Check if stationary long enough
                    elapsed_ms = (current_time_sec - stationary_start_time) * 1000
                    if elapsed_ms >= stationary_delay_ms:
                        # Stationary long enough - start streaming!
                        print(f"Stationary for {elapsed_ms:.0f}ms - starting stream at {arduino_send_fps} FPS")
                        is_streaming = True
                        waiting_for_stationary = False
                        stationary_start_time = None
                        # Send first frame immediately
                        if arduino.is_ready():
                            arduino.send_frame(binary)
                            last_send_time = current_time_sec
        
        else:
            # Idle - waiting for presence (enough white pixels)
            if has_presence:
                waiting_for_stationary = True
                stationary_start_time = None  # Don't start timer yet - wait for stillness
                print(f"Presence detected (white={white_ratio:.1%}), waiting for user to hold still...")
        
        prev_binary = binary.copy()  # Store for next frame's motion detection
        
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
