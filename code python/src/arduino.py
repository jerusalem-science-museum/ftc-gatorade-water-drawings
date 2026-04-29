"""
Arduino communication module for Water Drawing App.

Implements the Drop-Screen Arduino host API: Phase 1 params (once),
Phase 2 commands (s = send image then drop, d = drop buffer), ready byte 'r'.
Falls back to mock mode if no Arduino is connected.
No dependencies on other application modules.
"""

import time
from typing import Optional
import threading
import queue

import numpy as np
import serial
import serial.tools.list_ports

# Drop-Screen Arduino API constants (encode to bytes when writing/reading)
CMD_DROP = "d"  # drop current buffer
CMD_SEND_IMAGE = "s"  # send new image then drop
END_BYTE = "e"  # terminator after image bytes (firmware reads & checks)
READY_BYTE = "r"  # Arduino ready for next command
GO_BYTE = "g"  # after every 8 image bytes (flow control)
ARDUINO_WIDTH = 64  # Width fixed on Arduino
START_TIMEOUT = 2.0  # Seconds to wait for "START" after open


def find_arduino_port() -> Optional[str]:
    """
    Auto-detect Arduino serial port.

    Returns:
        Port name (e.g., 'COM3' or '/dev/ttyUSB0') if found, None otherwise.
    """
    # Common Arduino USB identifiers
    arduino_vids = [0x2341, 0x1A86, 0x0403, 0x10C4]  # Arduino, CH340, FTDI, CP210x

    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Check by VID
        if port.vid in arduino_vids:
            print(f"Found Arduino on {port.device} (VID: {hex(port.vid)})")
            return port.device
        # Check by description
        if port.description and any(
            x in port.description.lower() for x in ["arduino", "ch340", "usb serial"]
        ):
            print(f"Found Arduino on {port.device} ({port.description})")
            return port.device

    # List available ports for debugging
    if ports:
        print("Available serial ports:")
        for port in ports:
            print(
                f"  {port.device}: {port.description} (VID:{port.vid}, PID:{port.pid})"
            )
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
        mock: bool = False,
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
        self.ready = True
        self._params_sent = False
        self._go_queue = queue.Queue()  # one entry per 'g' received
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)

        # Force mock mode if requested
        if mock:
            print("[Arduino] Running in mock mode (forced)")
            self._mock = True

        # Auto-detect port if not specified
        if port is None:
            port = find_arduino_port()

        if port is None:
            print("[Arduino] No Arduino found - using mock mode")
            self._mock = True
            return

        # Connect and handshake
        try:
            self._serial = serial.Serial(port, baudrate)
            print(f"[Arduino] Connected to {port} at {baudrate} baud")
            # require getting start to start

            self.wait_for_start()
            self._reader_thread.start()
            self._send_session_params()

        except serial.SerialException as e:
            print(f"[Arduino] Failed to connect to {port}: {e}")
            print("[Arduino] Falling back to mock mode")
            self._mock = True

    def wait_for_start(self):
        """only start is the blocking msg we need, afterwards use _reader for msgs from arduino"""
        assert self._serial, "Problem initializing serial, exiting."
        msg = self._serial.readline().decode()
        assert "START" in msg, f"START not found in {msg}"
        print(f"[ARD->PI] {msg}")

    def _write_to_ard(self, msg):
        assert self._serial, "serial uninitialized"
        if isinstance(msg, str):
            print(f"[PI->ARD] {msg}")
            self._serial.write(msg.encode())
        elif isinstance(msg, bytes):
            print(".", end="")
            self._serial.write(msg)
        else:
            print(f"could write to ard {msg}")

    def _send_session_params(self) -> None:
        """Send Phase 1 params once: image_h, valve_on_time, drawing_depth (3 bytes)."""
        if self._serial is None or not self._serial.is_open or self._params_sent:
            return
        image_h = self.config["output_height"]
        valve_on_time = self.config["valve_on_time_ms"]
        drawing_depth = self.config["drawing_depth"]
        self._write_to_ard(bytes([image_h, valve_on_time, drawing_depth]))
        self._serial.flush()
        self._params_sent = True
        print(
            f"[Arduino] Sent params: image_h={image_h}, valve_on_time={valve_on_time}, drawing_depth={drawing_depth}"
        )

    @property
    def is_mock(self) -> bool:
        """Check if running in mock mode."""
        return self._mock

    def send_frame(self, binary_image: np.ndarray, cassette: int = 0) -> bool:
        """
        Send new image then drop. Wire sequence: 's' + cassette byte + image bytes + 'e'.
        Call only when ready is True.

        Args:
            binary_image: Binary (black/white) image, row-major, 64 columns expected.
            cassette: Which physical cassette to drop in. The firmware uses this
                to offset the column horizontally (cassette * image_w shift-register
                pulses). Range: 0 .. cassettes_num - 1.

        Returns:
            True if send was successful, False otherwise.
        """
        byte_array = self._pack_pixels(binary_image)
        h, w = binary_image.shape[0], binary_image.shape[1]

        if self._mock:
            white_pct = np.sum(binary_image) / (w * h * 255) * 100
            print(
                f"\n[MOCK] Send {len(byte_array)} bytes ({w}x{h}), "
                f"{white_pct:.1f}% white, cassette={cassette}"
            )
            self._print_ascii_preview(binary_image)

        if self._serial is None or not self._serial.is_open:
            return False

        try:
            # Drop any stale firmware output (parameter echoes, "drawing..." prints,
            # leftover flow-control bytes) before starting a fresh frame.
            self._serial.flush()
            self._serial.reset_input_buffer()

            # Frame header: 's' + cassette index byte
            self._write_to_ard(CMD_SEND_IMAGE.encode())
            self._write_to_ard(bytes([cassette & 0xFF]))

            # Image body in 8-byte chunks; drain any 'g' flow-control bytes per chunk.
            n = len(byte_array)
            for i in range(0, n, 8):
                self._write_to_ard(byte_array[i : i + 8])

                self._serial.flush()
                try:
                    self._go_queue.get(timeout=2.0)
                except queue.Empty:
                    print(
                        f"[Arduino] timeout waiting for 'g' after bytes {i}-{i+8}/{n}"
                    )
                    raise NotImplementedError  # or break, depending on how you want to handle it
                # print(f'bytes {i}-{i+8}/{n}')

            # Trailing END_KEY tells the firmware the image stream is complete.
            self.ready = (
                False  # make sure no race condition with arduino sending readykey.
            )
            self._write_to_ard(END_BYTE.encode())
            self._serial.flush()
            return True
        except serial.SerialException as e:
            print(f"[Arduino] Send error: {e}")
            return False

    def drop_current_buffer(self) -> bool:
        """
        Send 'd' to re-drop current buffer (no new image). Call only when self.ready is True.
        Returns:
            True if send was successful, False otherwise.
        """
        if self._mock:
            print("[MOCK] Drop current buffer")
            return True
        if self._serial is None or not self._serial.is_open:
            return False
        try:
            self._write_to_ard(CMD_DROP.encode())
            self._serial.flush()
            self.ready = False
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
            img = (
                img[:, :ARDUINO_WIDTH]
                if w > ARDUINO_WIDTH
                else np.pad(img, ((0, 0), (0, ARDUINO_WIDTH - w)), constant_values=0)
            )
        flat = (img.flatten() == 0).astype(np.uint8)
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
            line = "".join(["#" if p > 0 else " " for p in row[::col_step]])
            print(f"|{line}|")
        print("-" * (w // col_step + 2))

    def _reader(self):
        """Background thread — owns all serial reads."""
        while self._serial and self._serial.is_open:
            line = (
                self._serial.readline().decode().strip()
            )  # blocks; wakes on data or timeout
            if not line:
                continue
            elif line == READY_BYTE:
                self.ready = True
                print(f"[ARD->PI] {line}")
            elif line == GO_BYTE:
                self._go_queue.put(1)
                print(GO_BYTE,end='')
            else:
                print(f"[ARD->PI] {line}")
            
