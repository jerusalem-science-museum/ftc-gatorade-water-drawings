"""
Camera capture module for Water Drawing App.

Handles camera initialization with cross-platform support.
No dependencies on other application modules.
"""

import sys
from typing import Optional

import cv2


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
    Initialize video capture from camera with settings from config.
    
    Args:
        config: Configuration dictionary with video_source, capture_width, capture_height.
        
    Returns:
        cv2.VideoCapture object if successful, None otherwise.
    """
    camera_index = int(config["video_source"])
    
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
    
    # Report actual capture resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture initialized: {actual_w}x{actual_h}")
    
    return cap
