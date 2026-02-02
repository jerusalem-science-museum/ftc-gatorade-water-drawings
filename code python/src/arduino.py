"""
Arduino communication module for Water Drawing Calibration App.

This module provides a mock implementation that prints what would be sent.
Can be swapped for real serial implementation later.
"""

import numpy as np


class ArduinoSender:
    """Handles communication with Arduino for water drawing valve control."""
    
    def __init__(self, config: dict, mock: bool = True):
        """
        Initialize the Arduino sender.
        
        Args:
            config: Configuration dictionary with output_width, output_height, etc.
            mock: If True, just print what would be sent instead of actual serial.
        """
        self.mock = mock
        self.config = config
        self._ready = True
        
    def is_ready(self) -> bool:
        """
        Non-blocking check if Arduino can accept a new frame.
        
        Returns:
            True if Arduino is ready to receive, False otherwise.
        """
        if self.mock:
            return True  # Mock is always ready
        # Real implementation: check serial buffer for "done" signal
        return self._ready
        
    def send_frame(self, binary_image: np.ndarray) -> bool:
        """
        Convert binary image to byte array and send to Arduino (or mock print).
        
        Args:
            binary_image: Binary (black/white) image as numpy array.
            
        Returns:
            True if send was successful, False otherwise.
        """
        byte_array = self._pack_pixels(binary_image)
        w = self.config["output_width"]
        h = self.config["output_height"]
        
        if self.mock:
            white_pct = np.sum(binary_image) / (w * h * 255) * 100
            print(f"\n[MOCK ARDUINO] Would send {len(byte_array)} bytes ({w}x{h}), {white_pct:.1f}% white pixels")
            self._print_ascii_preview(binary_image)
            return True
        else:
            # Real implementation: non-blocking send, set self._ready = False
            # self._ready = False
            # self._serial.write(byte_array)
            pass
        
        return True
    
    def _pack_pixels(self, img: np.ndarray) -> bytes:
        """
        Pack 8 pixels into each byte (matches existing Arduino protocol).
        
        Args:
            img: Binary image where white pixels are > 0.
            
        Returns:
            Packed byte array.
        """
        flat = (img.flatten() > 0).astype(np.uint8)
        return bytes(np.packbits(flat))
    
    def _print_ascii_preview(self, img: np.ndarray) -> None:
        """
        Print a small ASCII art preview of the binary image.
        
        Args:
            img: Binary image to preview.
        """
        h, w = img.shape
        row_step = max(1, h // 10)  # Show ~10 rows
        col_step = max(1, w // 32)  # Show ~32 cols
        
        print("-" * (w // col_step + 2))
        for row in img[::row_step]:
            line = ''.join(['#' if p > 0 else ' ' for p in row[::col_step]])
            print(f"|{line}|")
        print("-" * (w // col_step + 2))
