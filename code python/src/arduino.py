"""
Arduino communication module for Water Drawing App.

Handles serial communication with Arduino for water valve control.
Falls back to mock mode if no Arduino is connected.
No dependencies on other application modules.
"""

import sys
from typing import Optional

import numpy as np

# Try to import serial, fall back gracefully if not installed
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. Running in mock mode only.")


def find_arduino_port() -> Optional[str]:
    """
    Auto-detect Arduino serial port.
    
    Returns:
        Port name (e.g., 'COM3' or '/dev/ttyUSB0') if found, None otherwise.
    """
    if not SERIAL_AVAILABLE:
        return None
    
    # Common Arduino USB identifiers
    arduino_vids = [0x2341, 0x1A86, 0x0403, 0x10C4]  # Arduino, CH340, FTDI, CP210x
    
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Check by VID
        if port.vid in arduino_vids:
            print(f"Found Arduino on {port.device} (VID: {hex(port.vid)})")
            return port.device
        # Check by description
        if port.description and any(x in port.description.lower() for x in ['arduino', 'ch340', 'usb serial']):
            print(f"Found Arduino on {port.device} ({port.description})")
            return port.device
    
    # List available ports for debugging
    if ports:
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device}: {port.description} (VID:{port.vid}, PID:{port.pid})")
    else:
        print("No serial ports found")
    
    return None


class ArduinoSender:
    """Handles communication with Arduino for water drawing valve control."""
    
    def __init__(
        self, 
        config: dict, 
        port: Optional[str] = None, 
        baudrate: int = 115200,
        mock: bool = False
    ):
        """
        Initialize the Arduino sender.
        
        Args:
            config: Configuration dictionary with output_width, output_height, etc.
            port: Serial port name (e.g., 'COM3'). If None, auto-detect.
            baudrate: Serial baudrate (default 115200).
            mock: If True, force mock mode (no serial communication).
        """
        self.config = config
        self._serial = None
        self._mock = False
        self._ready = True
        
        # Force mock mode if requested
        if mock:
            print("[Arduino] Running in mock mode (forced)")
            self._mock = True
            return
        
        if not SERIAL_AVAILABLE:
            print("[Arduino] pyserial not available - using mock mode")
            self._mock = True
            return
        
        # Auto-detect port if not specified
        if port is None:
            port = find_arduino_port()
        
        if port is None:
            print("[Arduino] No Arduino found - using mock mode")
            self._mock = True
            return
        
        # Try to connect
        try:
            self._serial = serial.Serial(port, baudrate, timeout=0.1)
            print(f"[Arduino] Connected to {port} at {baudrate} baud")
            self._mock = False
        except serial.SerialException as e:
            print(f"[Arduino] Failed to connect to {port}: {e}")
            print("[Arduino] Falling back to mock mode")
            self._mock = True
    
    @property
    def is_mock(self) -> bool:
        """Check if running in mock mode."""
        return self._mock
    
    def is_ready(self) -> bool:
        """
        Non-blocking check if Arduino can accept a new frame.
        
        Returns:
            True if Arduino is ready to receive, False otherwise.
        """
        if self._mock:
            return True
        
        if self._serial is None or not self._serial.is_open:
            return False
        
        # Check for acknowledgment from Arduino (non-blocking)
        if self._serial.in_waiting > 0:
            try:
                response = self._serial.readline().decode('utf-8', errors='ignore').strip()
                if response == "READY" or response == "OK":
                    self._ready = True
            except Exception:
                pass
        
        return self._ready
    
    def send_frame(self, binary_image: np.ndarray) -> bool:
        """
        Convert binary image to byte array and send to Arduino.
        
        Args:
            binary_image: Binary (black/white) image as numpy array.
            
        Returns:
            True if send was successful, False otherwise.
        """
        byte_array = self._pack_pixels(binary_image)
        w = self.config["output_width"]
        h = self.config["output_height"]
        
        if self._mock:
            white_pct = np.sum(binary_image) / (w * h * 255) * 100
            print(f"\n[MOCK] Send {len(byte_array)} bytes ({w}x{h}), {white_pct:.1f}% white")
            self._print_ascii_preview(binary_image)
            return True
        
        if self._serial is None or not self._serial.is_open:
            return False
        
        try:
            # Send start marker + data
            self._serial.write(b'\xAA')  # Start marker
            self._serial.write(byte_array)
            self._serial.write(b'\x55')  # End marker
            self._serial.flush()
            self._ready = False  # Wait for Arduino acknowledgment
            return True
        except serial.SerialException as e:
            print(f"[Arduino] Send error: {e}")
            return False
    
    def close(self) -> None:
        """Close the serial connection."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            print("[Arduino] Connection closed")
    
    def _pack_pixels(self, img: np.ndarray) -> bytes:
        """
        Pack 8 pixels into each byte (matches Arduino protocol).
        
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
        row_step = max(1, h // 10)
        col_step = max(1, w // 32)
        
        print("-" * (w // col_step + 2))
        for row in img[::row_step]:
            line = ''.join(['#' if p > 0 else ' ' for p in row[::col_step]])
            print(f"|{line}|")
        print("-" * (w // col_step + 2))
