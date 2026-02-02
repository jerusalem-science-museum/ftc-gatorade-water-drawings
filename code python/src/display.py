"""
Display module for Water Drawing App.

Handles display creation and overlay rendering.
No dependencies on other application modules.
"""

import cv2
import numpy as np


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
    
    # Build settings to display
    stationary_delay = config.get("stationary_delay_ms", 500)
    diff_mode = config.get("diff_mode", "both")
    morph_erode = config.get("morph_erode", 0)
    morph_dilate = config.get("morph_dilate", 0)
    
    lines = [
        f"Diff Threshold: {config['difference_threshold']} (+/- to adjust)",
        f"Diff Mode: {diff_mode} (d=cycle)",
        f"Morph: erode={morph_erode} dilate={morph_dilate} (e/E, l/L)",
        f"FPS: {fps:.1f}  White: {white_ratio * 100:.1f}%",
        f"Ref BG: {'SET' if has_reference_bg else 'NOT SET'} (r to capture)",
        f"Stationary delay: {stationary_delay}ms  {stationary_status}",
        f"Fullscreen: {'ON' if config['fullscreen'] else 'OFF'} (f)",
        "s=save, SPACE=send, q=quit"
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
