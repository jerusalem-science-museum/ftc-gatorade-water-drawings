"""
Arduino communication module for Water Drawing App.

Implements the Drop-Screen Arduino host API: Phase 1 params (once),
Phase 2 commands (s = send image then drop, d = drop buffer), ready byte 'r'.
Falls back to mock mode if no Arduino is connected.
No dependencies on other application modules.
"""

import time
from typing import Optional

import numpy as np

# Drop-Screen Arduino API constants (encode to bytes when writing/reading)
CMD_DROP = "d"  # drop current buffer
CMD_SEND_IMAGE = "s"  # send new image then drop
READY_BYTE = "r"  # Arduino ready for next command
GO_BYTE = "g"  # after every 8 image bytes (flow control)
ARDUINO_WIDTH = 64  # Width fixed on Arduino
START_TIMEOUT = 2.0  # Seconds to wait for "START" after open

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
        self._params_sent = False

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

        # Connect and handshake
        try:
            self._serial = serial.Serial(port, baudrate, timeout=0.1)
            print(f"[Arduino] Connected to {port} at {baudrate} baud")
            self._mock = False
            self._wait_for_start()
            self._send_session_params()
        except serial.SerialException as e:
            print(f"[Arduino] Failed to connect to {port}: {e}")
            print("[Arduino] Falling back to mock mode")
            self._mock = True

    def _wait_for_start(self) -> None:
        """Optionally wait for Arduino to send 'START' + newline after boot."""
        if self._serial is None or not self._serial.is_open:
            return
        deadline = time.monotonic() + START_TIMEOUT
        buf = b""
        while time.monotonic() < deadline:
            if self._serial.in_waiting > 0:
                buf += self._serial.read(self._serial.in_waiting)
                if b"START" in buf and (b"\n" in buf or b"\r" in buf):
                    print("[Arduino] Received START")
                    return
            time.sleep(0.02)
        # Timeout: continue anyway for older firmware that may not send START

    def _send_session_params(self) -> None:
        """Send Phase 1 params once: image_h, valve_on_time, drawing_depth (3 bytes)."""
        if self._serial is None or not self._serial.is_open or self._params_sent:
            return
        image_h = self.config["output_height"]
        valve_on_time = self.config["valve_on_time_ms"]
        drawing_depth = self.config["drawing_depth"]
        self._serial.write(bytes([image_h, valve_on_time, drawing_depth]))
        self._serial.flush()
        self._params_sent = True
        print(f"[Arduino] Sent params: image_h={image_h}, valve_on_time={valve_on_time}, drawing_depth={drawing_depth}")

    @property
    def is_mock(self) -> bool:
        """Check if running in mock mode."""
        return self._mock
    
    def is_ready(self) -> bool:
        """
        Non-blocking check if Arduino can accept a new frame.
        Reads bytes from serial; each 'r' (READY_BYTE) sets ready, 'g' is flow control.
        Returns:
            True if Arduino is ready to receive, False otherwise.
        """
        if self._mock:
            return True

        if self._serial is None or not self._serial.is_open:
            return False

        while self._serial.in_waiting > 0:
            b = self._serial.read(1)
            if not b:
                break
            if b == READY_BYTE.encode():
                self._ready = True
            # GO_BYTE (g) can be ignored or used for pacing; we just drain it

        return self._ready
    
    def send_frame(self, binary_image: np.ndarray) -> bool:
        """
        Send new image then drop (API command 's'): one byte 's' then image_h×8 bytes.
        Call only when is_ready() is True.
        Args:
            binary_image: Binary (black/white) image, row-major, 64 columns expected.
        Returns:
            True if send was successful, False otherwise.
        """
        byte_array = self._pack_pixels(binary_image)
        h, w = binary_image.shape[0], binary_image.shape[1]

        if self._mock:
            white_pct = np.sum(binary_image) / (w * h * 255) * 100
            print(f"\n[MOCK] Send {len(byte_array)} bytes ({w}x{h}), {white_pct:.1f}% white")
            self._print_ascii_preview(binary_image)
            return True

        if self._serial is None or not self._serial.is_open:
            return False

        try:
            self._serial.write(CMD_SEND_IMAGE.encode())
            # Send image in chunks of 8 bytes; optionally drain one 'g' per chunk
            n = len(byte_array)
            for i in range(0, n, 8):
                chunk = byte_array[i : i + 8]
                self._serial.write(chunk)
                self._serial.flush()
                # Optionally wait for 'g' (flow control) - non-blocking drain
                if self._serial.in_waiting > 0:
                    self._serial.read(1)
            self._ready = False
            return True
        except serial.SerialException as e:
            print(f"[Arduino] Send error: {e}")
            return False

    def drop_current_buffer(self) -> bool:
        """
        Send 'd' to re-drop current buffer (no new image). Call only when is_ready() is True.
        Returns:
            True if send was successful, False otherwise.
        """
        if self._mock:
            print("[MOCK] Drop current buffer")
            return True
        if self._serial is None or not self._serial.is_open:
            return False
        try:
            self._serial.write(CMD_DROP.encode())
            self._serial.flush()
            self._ready = False
            return True
        except serial.SerialException as e:
            print(f"[Arduino] Drop error: {e}")
            return False
    
    def close(self) -> None:
        """Close the serial connection."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            print("[Arduino] Connection closed")
    
    def _pack_pixels(self, img: np.ndarray) -> bytes:
        """
        Pack image to row-major bytes: 8 bytes per row (64 px), MSB first per byte.
        API: bit 7 = first pixel of 8, bit 0 = last; 1 = valve on, 0 = off.
        Uses np.packbits(..., bitorder='big') for MSB-first.
        """
        h, w = img.shape[0], img.shape[1]
        if w != ARDUINO_WIDTH:
            img = img[:, :ARDUINO_WIDTH] if w > ARDUINO_WIDTH else np.pad(
                img, ((0, 0), (0, ARDUINO_WIDTH - w)), constant_values=0
            )
        flat = (img.flatten() > 0).astype(np.uint8)
        packed = np.packbits(flat, bitorder="big")
        return packed.tobytes()
    
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
