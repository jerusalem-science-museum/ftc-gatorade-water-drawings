"""
Water Drawing App - Main Entry Point

Thin orchestrator: capture frame -> process -> compute metrics -> display
-> streaming policy -> key handling. Display owns the window/overlay/FPS;
StreamingController owns the state machine and the Arduino.
"""

import sys
import time
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from config import get_config_path, load_config, save_config
from capture import init_capture
from processing import process_frame, capture_reference_background
from display import Display
from streaming import FrameMetrics, StreamingController


class RefAction(Enum):
    NONE = 0
    CAPTURE = 1


class WaterDrawingApp:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.cap = init_capture(self.config)
        if self.cap is None:
            raise RuntimeError("Failed to initialize video capture.")

        self.display = Display(self.config)
        self.streaming = StreamingController(self.config, config_path)
        self.reference_bg: Optional[np.ndarray] = None

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

    def _compute_frame_metrics(self, binary: np.ndarray) -> FrameMetrics:
        cfg = self.config
        prev_binary = self.streaming.state.prev_binary
        total_pixels = cfg["output_width"] * cfg["output_height"]
        white_ratio = float(np.sum(binary)) / (total_pixels * 255)
        stationary_threshold = cfg["stationary_threshold"]
        if prev_binary is not None:
            pixel_diff = cv2.bitwise_xor(binary, prev_binary)
            changed_pixels = float(np.sum(pixel_diff)) / 255
            pixel_change_ratio = changed_pixels / total_pixels
            is_stationary = pixel_change_ratio < stationary_threshold
        else:
            pixel_change_ratio = 0.0
            is_stationary = False
        return FrameMetrics(
            binary=binary,
            white_ratio=white_ratio,
            pixel_change_ratio=pixel_change_ratio,
            is_stationary=is_stationary,
            has_presence=white_ratio > cfg["min_presence_threshold"],
            current_time_sec=time.time(),
        )

    def _handle_key(self, key: int, binary: np.ndarray) -> tuple[bool, RefAction]:
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
            self.display.toggle_fullscreen()
        elif key == ord('d'):
            self.display.toggle_overlay()
        elif key == ord('m'):
            modes = ["both", "lighter", "darker"]
            idx = (modes.index(cfg["diff_mode"]) + 1) % len(modes)
            cfg["diff_mode"] = modes[idx]
            print(f"Diff mode: {cfg['diff_mode']}")
        elif key == ord('e'):
            cfg["morph_erode"] += 1
            print(f"Morph erode: {cfg['morph_erode']}")
        elif key == ord('E'):
            cfg["morph_erode"] = max(0, cfg["morph_erode"] - 1)
            print(f"Morph erode: {cfg['morph_erode']}")
        elif key == ord('l'):
            cfg["morph_dilate"] += 1
            print(f"Morph dilate: {cfg['morph_dilate']}")
        elif key == ord('L'):
            cfg["morph_dilate"] = max(0, cfg["morph_dilate"] - 1)
            print(f"Morph dilate: {cfg['morph_dilate']}")
        elif key == ord(' '):
            self.streaming.manual_send(binary)
        return True, RefAction.NONE

    def run(self) -> None:
        running = True
        while running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            _small, binary, gray = process_frame(frame, self.config, self.reference_bg)
            metrics = self._compute_frame_metrics(binary)

            self.display.render(
                frame, metrics, self.streaming.state,
                reference_bg_set=self.reference_bg is not None,
            )
            self.streaming.tick(metrics)
            self.display.update_fps()

            key = cv2.waitKey(1) & 0xFF
            running, ref_action = self._handle_key(key, metrics.binary)
            if ref_action == RefAction.CAPTURE:
                self.reference_bg = gray.copy()
                print("Reference background captured!")

        self.cap.release()
        self.streaming.close()
        self.display.close()
        print("Application closed.")


def main() -> None:
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
