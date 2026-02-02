"""
Water Drawing App - Main Entry Point

Orchestrates camera capture, image processing, display, and Arduino communication.
Each module is independent and can fail without affecting others.

Key design: Display loop is decoupled from Arduino communication.
Display NEVER waits for Arduino - user always sees real-time video.
"""

import subprocess
import sys
import time
from typing import Optional

import cv2
import numpy as np


def disable_screen_blanking():
    """Disable screen blanking/DPMS on Linux to prevent display from fading to black."""
    if not sys.platform.startswith('linux'):
        return
    
    print("Disabling screen blanking...")
    commands = [
        # Disable DPMS (Display Power Management Signaling)
        "xset -dpms",
        # Disable screen saver
        "xset s off",
        # Disable screen blanking
        "xset s noblank",
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=2)
            if result.returncode == 0:
                print(f"  OK: {cmd}")
            else:
                print(f"  SKIP: {cmd} (may need X display)")
        except Exception as e:
            print(f"  SKIP: {cmd} ({e})")

# Import application modules
from config import get_config_path, load_config, save_config
from capture import init_capture
from processing import process_frame, capture_reference_background
from display import create_stacked_display, draw_overlay, set_fullscreen
from arduino import ArduinoSender


def main():
    """Main entry point for the water drawing app."""
    
    # Disable screen blanking on Linux (prevents fade to black)
    disable_screen_blanking()
    
    # Load configuration
    config_path = get_config_path()
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Initialize video capture
    cap = init_capture(config)
    if cap is None:
        print("Failed to initialize video capture. Exiting.")
        sys.exit(1)
    
    # Initialize Arduino sender
    arduino_port = config.get("arduino_port")  # None = auto-detect
    arduino_mock = config.get("arduino_mock", False)  # Force mock mode
    arduino = ArduinoSender(config, port=arduino_port, mock=arduino_mock)
    
    # Create window
    window_name = "Water Drawing"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set initial fullscreen state
    if config.get("fullscreen", False):
        set_fullscreen(window_name, True)
    
    # State variables
    frame_count = 0
    fps = 0.0
    fps_update_time = cv2.getTickCount()
    
    # Static background state
    reference_bg: Optional[np.ndarray] = None
    
    # Streaming state
    waiting_for_stationary = False
    stationary_start_time: Optional[float] = None
    is_streaming = False
    last_send_time: float = 0.0
    prev_binary: Optional[np.ndarray] = None
    
    print("\n=== Water Drawing App ===")
    print("Controls:")
    print("  +/- or LEFT/RIGHT: Adjust difference threshold")
    print("  d: Cycle diff mode (both/lighter/darker)")
    print("  e/E: Increase/decrease erode iterations")
    print("  l/L: Increase/decrease dilate iterations")
    print("  r: Capture reference background")
    print("  s: Save config")
    print("  f: Toggle fullscreen")
    print("  SPACE: Manual send to Arduino")
    print("  q/ESC: Quit")
    print("=========================\n")
    
    # Capture initial reference background after a short delay
    print("Capturing initial reference background in 2 seconds...")
    print("Please ensure the scene is EMPTY (no hands)!")
    time.sleep(2)
    
    # Read a few frames to stabilize camera
    for _ in range(10):
        cap.read()
    
    ret, init_frame = cap.read()
    if ret:
        reference_bg = capture_reference_background(init_frame, config)
        print("Initial reference background captured!")
    else:
        print("Warning: Could not capture initial reference background")
    
    # Track frame brightness to detect camera going dark
    last_brightness_warning = 0
    consecutive_dark_frames = 0
    
    running = True
    while running:
        # -------- CAPTURE --------
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Check if frame is very dark (camera might be auto-exposing incorrectly)
        frame_brightness = np.mean(frame)
        if frame_brightness < 10:  # Very dark frame
            consecutive_dark_frames += 1
            if consecutive_dark_frames > 30 and time.time() - last_brightness_warning > 5:
                print(f"WARNING: Camera producing dark frames (brightness={frame_brightness:.1f})")
                print("  - Check camera exposure settings")
                print("  - Camera may need re-initialization")
                last_brightness_warning = time.time()
        else:
            consecutive_dark_frames = 0
        
        # -------- PROCESS --------
        small, binary, gray = process_frame(frame, config, reference_bg)
        
        # -------- DISPLAY --------
        # Apply flips to full-res frame for high-res preview
        display_original = frame.copy()
        if config.get("flip_vertical", False):
            display_original = cv2.flip(display_original, 0)
        if config.get("flip_horizontal", False):
            display_original = cv2.flip(display_original, 1)
        
        display_frame = create_stacked_display(display_original, binary, config["display_scale"])
        
        # Calculate white ratio for presence detection
        total_pixels = config["output_width"] * config["output_height"]
        white_ratio = np.sum(binary) / (total_pixels * 255)
        
        # Calculate pixel movement using XOR comparison
        stationary_threshold = config.get("stationary_threshold", 0.02)
        if prev_binary is not None:
            pixel_diff = cv2.bitwise_xor(binary, prev_binary)
            changed_pixels = np.sum(pixel_diff) / 255
            pixel_change_ratio = changed_pixels / total_pixels
            is_currently_still = pixel_change_ratio < stationary_threshold
        else:
            pixel_change_ratio = 0.0
            is_currently_still = False
        
        # Build status string for overlay
        require_stationary = config.get("require_stationary_for_send", True)
        arduino_send_fps = config.get("arduino_send_fps", 10)
        send_interval = 1.0 / arduino_send_fps if arduino_send_fps > 0 else 0
        
        if is_streaming:
            if require_stationary:
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
        
        # -------- PRESENCE DETECTION & STREAMING --------
        stationary_delay_ms = config.get("stationary_delay_ms", 500)
        min_presence = config.get("min_presence_threshold", 0.05)
        
        has_presence = white_ratio > min_presence
        is_stationary = is_currently_still
        current_time_sec = time.time()
        
        if is_streaming:
            if not has_presence:
                is_streaming = False
                stationary_start_time = None
                print("Presence ended, stopping stream")
            elif require_stationary:
                time_since_last_send = current_time_sec - last_send_time
                fps_ready = time_since_last_send >= send_interval
                
                if not is_stationary:
                    if stationary_start_time is not None:
                        print(f"Movement during stream (pixel change={pixel_change_ratio:.1%}), waiting...")
                    stationary_start_time = None
                else:
                    if stationary_start_time is None:
                        stationary_start_time = current_time_sec
                    
                    elapsed_stationary_ms = (current_time_sec - stationary_start_time) * 1000
                    stationary_ready = elapsed_stationary_ms >= stationary_delay_ms
                    
                    if stationary_ready and fps_ready and arduino.is_ready():
                        arduino.send_frame(binary)
                        last_send_time = current_time_sec
                        stationary_start_time = None
                        print(f"Sent frame (stationary for {elapsed_stationary_ms:.0f}ms)")
            else:
                if arduino.is_ready() and (current_time_sec - last_send_time) >= send_interval:
                    arduino.send_frame(binary)
                    last_send_time = current_time_sec
        
        elif waiting_for_stationary:
            if not has_presence:
                waiting_for_stationary = False
                stationary_start_time = None
                print("Presence ended before stationary")
            elif not is_stationary:
                if stationary_start_time is not None:
                    print(f"Movement detected (pixel change={pixel_change_ratio:.1%}), resetting...")
                stationary_start_time = None
            else:
                if stationary_start_time is None:
                    stationary_start_time = current_time_sec
                    print(f"User stationary, starting {stationary_delay_ms}ms timer...")
                else:
                    elapsed_ms = (current_time_sec - stationary_start_time) * 1000
                    if elapsed_ms >= stationary_delay_ms:
                        print(f"Stationary for {elapsed_ms:.0f}ms - starting stream")
                        is_streaming = True
                        waiting_for_stationary = False
                        stationary_start_time = None
                        if arduino.is_ready():
                            arduino.send_frame(binary)
                            last_send_time = current_time_sec
        
        else:
            if has_presence:
                waiting_for_stationary = True
                stationary_start_time = None
                print(f"Presence detected (white={white_ratio:.1%}), waiting for stillness...")
        
        prev_binary = binary.copy()
        
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
            step = config.get("difference_threshold_step", 5)
            config["difference_threshold"] = min(255, config["difference_threshold"] + step)
            print(f"Difference threshold: {config['difference_threshold']}")
            
        elif key in [ord('-'), 81, 1]:  # - or left arrow
            step = config.get("difference_threshold_step", 5)
            config["difference_threshold"] = max(0, config["difference_threshold"] - step)
            print(f"Difference threshold: {config['difference_threshold']}")
            
        elif key == ord('s'):
            save_config(config_path, config)
            
        elif key == ord('r'):
            # Capture new reference background
            reference_bg = gray.copy()
            print("Reference background captured!")
            
        elif key == ord('f'):
            config["fullscreen"] = not config.get("fullscreen", False)
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
            config["morph_erode"] = config.get("morph_erode", 0) + 1
            print(f"Morph erode: {config['morph_erode']}")
        
        elif key == ord('E'):
            config["morph_erode"] = max(0, config.get("morph_erode", 0) - 1)
            print(f"Morph erode: {config['morph_erode']}")
        
        elif key == ord('l'):
            config["morph_dilate"] = config.get("morph_dilate", 0) + 1
            print(f"Morph dilate: {config['morph_dilate']}")
        
        elif key == ord('L'):
            config["morph_dilate"] = max(0, config.get("morph_dilate", 0) - 1)
            print(f"Morph dilate: {config['morph_dilate']}")
            
        elif key == ord(' '):  # SPACE - manual send
            print("Manual send triggered")
            arduino.send_frame(binary)
    
    # Cleanup
    cap.release()
    arduino.close()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
