"""
Water Drawing App - Main Entry Point

Orchestrates camera capture, image processing, display, and Arduino communication.
Each module is independent and can fail without affecting others.

Key design: Display loop is decoupled from Arduino communication.
Display NEVER waits for Arduino - user always sees real-time video.
"""

import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

# Import application modules
from config import get_config_path, load_config, save_config
from capture import init_capture, set_camera_controls_linux
from processing import process_frame, capture_reference_background
from display import create_stacked_display, draw_overlay, set_fullscreen
from arduino import ArduinoSender


# ---------------------------------------------------------------------------
# Loop state (bundled to shorten param lists)
# ---------------------------------------------------------------------------


class RefAction(Enum):
    """Result of key handling: no ref change, or capture new reference background."""
    NONE = 0
    CAPTURE = 1


@dataclass
class StreamingState:
    """Presence/stationary/streaming state carried across frames."""
    waiting_for_stationary: bool
    is_streaming: bool
    stationary_start_time: Optional[float]
    last_send_time: float
    prev_binary: Optional[np.ndarray]


@dataclass
class FpsState:
    """FPS counter state."""
    frame_count: int
    fps: float
    fps_update_time: float


# DisplayContext removed: class holds config, overlay_visible, fps_state, reference_bg, window_name


# ---------------------------------------------------------------------------
# WaterDrawingApp: owns all state and environment; methods take per-frame data only
# ---------------------------------------------------------------------------


class WaterDrawingApp:
    """
    Single app instance: config, capture, arduino, window, and loop state.
    run() executes the main loop; methods use self for config/state and take frame/event data.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.cap = init_capture(self.config)
        if self.cap is None:
            raise RuntimeError("Failed to initialize video capture.")
        self.arduino = ArduinoSender(
            self.config,
            port=self.config["arduino_port"],
            mock=self.config["arduino_mock"],
        )
        self.window_name = "Water Drawing"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.config["fullscreen"]:
            set_fullscreen(self.window_name, True)

        self.fps_state = FpsState(
            frame_count=0, fps=0.0, fps_update_time=cv2.getTickCount()
        )
        self.reference_bg: Optional[np.ndarray] = None
        self.streaming_state = StreamingState(
            waiting_for_stationary=False,
            is_streaming=False,
            stationary_start_time=None,
            last_send_time=0.0,
            prev_binary=None,
        )
        self.overlay_visible = False

        self._print_controls()
        self._capture_initial_reference()

    def _print_controls(self) -> None:
        print("\n=== Water Drawing App ===")
        print("Controls:")
        print("  +/- or LEFT/RIGHT: Adjust difference threshold")
        print("  m: Cycle diff mode (both/lighter/darker)")
        print("  e/E: Increase/decrease erode iterations")
        print("  l/L: Increase/decrease dilate iterations")
        print("  r: Capture reference background")
        print("  s: Save config")
        print("  d: Toggle overlay display")
        print("  f: Toggle fullscreen")
        print("  SPACE: Manual send to Arduino")
        print("  q/ESC: Quit")
        print("=========================\n")

    def _capture_initial_reference(self) -> None:
        print("Capturing initial reference background in 2 seconds...")
        print("Please ensure the scene is EMPTY (no hands)!")
        time.sleep(2)
        for _ in range(10):
            self.cap.read()
        ret, init_frame = self.cap.read()
        if ret:
            self.reference_bg = capture_reference_background(init_frame, self.config)
            print("Initial reference background captured!")
        else:
            print("Warning: Could not capture initial reference background")

    def _capture_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one frame. Returns (success, frame)."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def _compute_display_metrics(self, binary, prev_binary):
        """Compute white ratio and pixel movement. Returns (white_ratio, pixel_change_ratio, is_currently_still)."""
        cfg = self.config
        total_pixels = cfg["output_width"] * cfg["output_height"]
        white_ratio = np.sum(binary) / (total_pixels * 255)
        stationary_threshold = cfg["stationary_threshold"]
        if prev_binary is not None:
            pixel_diff = cv2.bitwise_xor(binary, prev_binary)
            changed_pixels = np.sum(pixel_diff) / 255
            pixel_change_ratio = changed_pixels / total_pixels
            is_currently_still = pixel_change_ratio < stationary_threshold
        else:
            pixel_change_ratio = 0.0
            is_currently_still = False
        return white_ratio, pixel_change_ratio, is_currently_still

    def _get_stationary_status_string(self, pixel_change_ratio: float) -> str:
        """Build the overlay status string for streaming/stationary state."""
        s = self.streaming_state
        cfg = self.config
        stationary_threshold = cfg["stationary_threshold"]
        require_stationary = cfg["require_stationary_for_send"]
        arduino_send_fps = cfg["arduino_send_fps"]
        send_interval = 1.0 / arduino_send_fps if arduino_send_fps > 0 else 0

        if s.is_streaming:
            if require_stationary:
                time_since_send = time.time() - s.last_send_time
                fps_ready = time_since_send >= send_interval
                fps_status = "ready" if fps_ready else f"wait {(send_interval - time_since_send):.1f}s"
                if s.stationary_start_time is not None:
                    elapsed_ms = (time.time() - s.stationary_start_time) * 1000
                    return f"[HOLD: {elapsed_ms:.0f}ms | FPS: {fps_status}] move:{pixel_change_ratio:.1%}"
                return f"[MOVING | FPS: {fps_status}] move:{pixel_change_ratio:.1%}"
            return f"[STREAMING @ {arduino_send_fps} FPS] move:{pixel_change_ratio:.1%}"
        if s.waiting_for_stationary:
            if s.stationary_start_time is not None:
                elapsed_ms = (time.time() - s.stationary_start_time) * 1000
                return f"[HOLD STILL: {elapsed_ms:.0f}ms] move:{pixel_change_ratio:.1%}"
            return f"[MOVING {pixel_change_ratio:.1%} > {stationary_threshold:.1%}]"
        return ""

    def _render_display(self, frame, binary, white_ratio: float, stationary_status: str) -> None:
        """Apply flips, build stacked view, draw overlay if enabled, show frame."""
        cfg = self.config
        display_original = frame.copy()
        if cfg["flip_vertical"]:
            display_original = cv2.flip(display_original, 0)
        if cfg["flip_horizontal"]:
            display_original = cv2.flip(display_original, 1)
        display_frame = create_stacked_display(
            display_original, binary, cfg["display_scale"]
        )
        if self.overlay_visible:
            display_frame = draw_overlay(
                display_frame, cfg, self.fps_state.fps, white_ratio,
                has_reference_bg=(self.reference_bg is not None),
                stationary_status=stationary_status
            )
        cv2.imshow(self.window_name, display_frame)


    def _update_streaming_state(
        self,
        has_presence: bool,
        is_stationary: bool,
        current_time_sec: float,
        binary,
        white_ratio: float,
        pixel_change_ratio: float,
    ) -> None:
        """One step of the presence/stationary/streaming state machine. Caller sets .prev_binary after."""
        cfg = self.config
        ard = self.arduino
        s = self.streaming_state
        stationary_delay_ms = cfg["stationary_delay_ms"]
        require_stationary = cfg["require_stationary_for_send"]
        arduino_send_fps = cfg["arduino_send_fps"]
        send_interval = 1.0 / arduino_send_fps if arduino_send_fps > 0 else 0

        waiting = s.waiting_for_stationary
        streaming = s.is_streaming
        start_time = s.stationary_start_time
        last_send = s.last_send_time

        if streaming:
            if not has_presence:
                streaming = False
                start_time = None
                print("Presence ended, stopping stream")
            elif require_stationary:
                time_since_last_send = current_time_sec - last_send
                fps_ready = time_since_last_send >= send_interval
                if not is_stationary:
                    if start_time is not None:
                        print(
                            f"Movement during stream (pixel change={pixel_change_ratio:.1%}), waiting..."
                        )
                    start_time = None
                else:
                    if start_time is None:
                        start_time = current_time_sec
                    elapsed_stationary_ms = (current_time_sec - start_time) * 1000
                    stationary_ready = elapsed_stationary_ms >= stationary_delay_ms
                    if stationary_ready and fps_ready and ard.is_ready():
                        ard.send_frame(binary)
                        last_send = current_time_sec
                        start_time = None
                        print(f"Sent frame (stationary for {elapsed_stationary_ms:.0f}ms)")
            else:
                if ard.is_ready() and (current_time_sec - last_send) >= send_interval:
                    ard.send_frame(binary)
                    last_send = current_time_sec

        elif waiting:
            if not has_presence:
                waiting = False
                start_time = None
                print("Presence ended before stationary")
            elif not is_stationary:
                if start_time is not None:
                    print(
                        f"Movement detected (pixel change={pixel_change_ratio:.1%}), resetting..."
                    )
                start_time = None
            else:
                if start_time is None:
                    start_time = current_time_sec
                    print(f"User stationary, starting {stationary_delay_ms}ms timer...")
                else:
                    elapsed_ms = (current_time_sec - start_time) * 1000
                    if elapsed_ms >= stationary_delay_ms:
                        print(f"Stationary for {elapsed_ms:.0f}ms - starting stream")
                        streaming = True
                        waiting = False
                        start_time = None
                        if ard.is_ready():
                            ard.send_frame(binary)
                            last_send = current_time_sec

        else:
            if has_presence:
                waiting = True
                start_time = None
                print(f"Presence detected (white={white_ratio:.1%}), waiting for stillness...")

        self.streaming_state = StreamingState(
            waiting_for_stationary=waiting,
            is_streaming=streaming,
            stationary_start_time=start_time,
            last_send_time=last_send,
            prev_binary=s.prev_binary,
        )


    def _update_fps(self) -> None:
        """Update FPS every second; mutates self.fps_state."""
        s = self.fps_state
        count = s.frame_count + 1
        current_time = cv2.getTickCount()
        elapsed = (current_time - s.fps_update_time) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            self.fps_state = FpsState(frame_count=0, fps=count / elapsed, fps_update_time=current_time)
        else:
            self.fps_state = FpsState(frame_count=count, fps=s.fps, fps_update_time=s.fps_update_time)

    def _handle_key(self, key: int, binary) -> tuple[bool, RefAction]:
        """Handle one key press. Mutates config and self.overlay_visible. Returns (running, ref_action)."""
        cfg = self.config
        if key == ord('q') or key == 27:
            return False, RefAction.NONE

        if key in [ord('+'), ord('='), 83, 0]:
            step = cfg["difference_threshold_step"]
            cfg["difference_threshold"] = min(255, cfg["difference_threshold"] + step)
            print(f"Difference threshold: {cfg['difference_threshold']}")
        elif key in [ord('-'), 81, 1]:
            step = cfg["difference_threshold_step"]
            cfg["difference_threshold"] = max(0, cfg["difference_threshold"] - step)
            print(f"Difference threshold: {cfg['difference_threshold']}")
        elif key == ord('s'):
            save_config(self.config_path, cfg)
        elif key == ord('r'):
            return True, RefAction.CAPTURE
        elif key == ord('f'):
            cfg["fullscreen"] = not cfg["fullscreen"]
            set_fullscreen(self.window_name, cfg["fullscreen"])
            print(f"Fullscreen: {'ON' if cfg['fullscreen'] else 'OFF'}")
        elif key == ord('d'):
            self.overlay_visible = not self.overlay_visible
            print(f"Overlay: {'ON' if self.overlay_visible else 'OFF'}")
        elif key == ord('m'):
            modes = ["both", "lighter", "darker"]
            current = cfg["diff_mode"]
            idx = (modes.index(current) + 1) % len(modes)
            cfg["diff_mode"] = modes[idx]
            print(f"Diff mode: {cfg['diff_mode']}")
        elif key == ord('e'):
            cfg["morph_erode"] = cfg["morph_erode"] + 1
            print(f"Morph erode: {cfg['morph_erode']}")
        elif key == ord('E'):
            cfg["morph_erode"] = max(0, cfg["morph_erode"] - 1)
            print(f"Morph erode: {cfg['morph_erode']}")
        elif key == ord('l'):
            cfg["morph_dilate"] = cfg["morph_dilate"] + 1
            print(f"Morph dilate: {cfg['morph_dilate']}")
        elif key == ord('L'):
            cfg["morph_dilate"] = max(0, cfg["morph_dilate"] - 1)
            print(f"Morph dilate: {cfg['morph_dilate']}")
        elif key == ord(' '):
            print("Manual send triggered")
            self.arduino.send_frame(binary)
        return True, RefAction.NONE


    def run(self) -> None:
        """Main loop: capture, process, display, streaming, FPS, keys. Cleans up on exit."""
        running = True
        while running:
            ok, frame = self._capture_frame()
            if not ok:
                print("Error: Could not read frame")
                break

            small, binary, gray = process_frame(frame, self.config, self.reference_bg)

            white_ratio, pixel_change_ratio, is_currently_still = self._compute_display_metrics(
                binary, self.streaming_state.prev_binary
            )
            stationary_status = self._get_stationary_status_string(pixel_change_ratio)

            self._render_display(frame, binary, white_ratio, stationary_status)

            min_presence = self.config["min_presence_threshold"]
            has_presence = white_ratio > min_presence
            is_stationary = is_currently_still
            current_time_sec = time.time()
            self._update_streaming_state(
                has_presence, is_stationary, current_time_sec, binary,
                white_ratio, pixel_change_ratio
            )
            self.streaming_state.prev_binary = binary.copy()

            self._update_fps()

            key = cv2.waitKey(1) & 0xFF
            running, ref_action = self._handle_key(key, binary)
            if ref_action == RefAction.CAPTURE:
                self.reference_bg = gray.copy()
                print("Reference background captured!")

        self.cap.release()
        self.arduino.close()
        cv2.destroyAllWindows()
        print("Application closed.")


def main() -> None:
    """Load config, create app, run. Exits on config or capture init failure."""
    config_path = get_config_path()
    try:
        app = WaterDrawingApp(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
