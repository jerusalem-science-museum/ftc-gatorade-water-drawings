"""
Streaming policy for Water Drawing App.

Owns the presence -> stationary -> streaming state machine, idle PNG
rotation, and the ArduinoSender. Decides *when* to send to the hardware;
the protocol layer (arduino.py) decides *how*.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from arduino import ArduinoSender
from processing import preprocess_idle_image


@dataclass
class FrameMetrics:
    """Per-frame computed values, bundled so signatures stop growing."""
    binary: np.ndarray
    white_ratio: float
    pixel_change_ratio: float
    is_stationary: bool
    has_presence: bool
    current_time_sec: float


@dataclass
class StreamingState:
    """Presence/stationary/streaming state carried across frames."""
    waiting_for_stationary: bool
    is_streaming: bool
    stationary_start_time: Optional[float]
    last_send_time: float
    prev_binary: Optional[np.ndarray]


class StreamingController:
    """
    Policy layer: decides when to send frames vs. drop the buffer vs. show
    idle PNGs. Owns the ArduinoSender.
    """

    def __init__(self, config: dict, config_path: str):
        self.config = config
        self.config_path = config_path
        self._arduino = ArduinoSender(
            config,
            port=config["arduino_port"],
            mock=config["arduino_mock"],
        )
        self._state = StreamingState(
            waiting_for_stationary=False,
            is_streaming=False,
            stationary_start_time=None,
            last_send_time=0.0,
            prev_binary=None,
        )
        self.empty_streak = 0
        self.idle_index = 0
        self.idle_images: list[str] = self._load_idle_images()

    @property
    def state(self) -> StreamingState:
        return self._state

    def close(self) -> None:
        self._arduino.close()

    def manual_send(self, binary: np.ndarray) -> None:
        print("Manual send triggered")
        self._arduino.send_frame(binary, cassette=self._random_cassette())

    def tick(self, metrics: FrameMetrics) -> None:
        """One per-frame step: state machine + idle behavior + prev_binary update."""
        if metrics.has_presence:
            self.empty_streak = 0

        self._update_streaming_state(metrics)

        if (
            not metrics.has_presence
            and not self._state.is_streaming
            and not self._state.waiting_for_stationary
        ):
            self._handle_empty_frame(metrics.current_time_sec)

        self._state.prev_binary = metrics.binary.copy()

    def _load_idle_images(self) -> list[str]:
        rel_dir = self.config.get("idle_images_dir")
        if not rel_dir:
            return []
        base = os.path.dirname(os.path.abspath(self.config_path))
        idle_dir = os.path.join(base, rel_dir)
        if not os.path.isdir(idle_dir):
            print(f"[Idle] Directory not found: {idle_dir} (idle PNG rotation disabled)")
            return []
        exts = (".png", ".jpg", ".jpeg")
        files = sorted(
            os.path.join(idle_dir, f)
            for f in os.listdir(idle_dir)
            if f.lower().endswith(exts)
        )
        print(f"[Idle] Loaded {len(files)} idle images from {idle_dir}")
        return files

    def _random_cassette(self) -> int:
        n = max(1, self.config.get("cassettes_num", 1))
        return random.randint(0, n - 1)

    def _update_streaming_state(self, metrics: FrameMetrics) -> None:
        cfg = self.config
        ard = self._arduino
        s = self._state
        stationary_delay_ms = cfg["stationary_delay_ms"]
        require_stationary = cfg["require_stationary_for_send"]
        arduino_send_fps = cfg["arduino_send_fps"]
        send_interval = 1.0 / arduino_send_fps if arduino_send_fps > 0 else 0

        waiting = s.waiting_for_stationary
        streaming = s.is_streaming
        start_time = s.stationary_start_time
        last_send = s.last_send_time

        has_presence = metrics.has_presence
        is_stationary = metrics.is_stationary
        current_time_sec = metrics.current_time_sec
        binary = metrics.binary
        white_ratio = metrics.white_ratio
        pixel_change_ratio = metrics.pixel_change_ratio

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
                    if stationary_ready and fps_ready and ard.ready:
                        ard.send_frame(binary, cassette=self._random_cassette())
                        last_send = current_time_sec
                        start_time = None
                        print(f"Sent frame (stationary for {elapsed_stationary_ms:.0f}ms)")
            else:
                if ard.ready and (current_time_sec - last_send) >= send_interval:
                    ard.send_frame(binary, cassette=self._random_cassette())
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
                        if ard.ready:
                            ard.send_frame(binary, cassette=self._random_cassette())
                            last_send = current_time_sec

        else:
            if has_presence:
                waiting = True
                start_time = None
                print(f"Presence detected (white={white_ratio:.1%}), waiting for stillness...")

        s.waiting_for_stationary = waiting
        s.is_streaming = streaming
        s.stationary_start_time = start_time
        s.last_send_time = last_send

    def _handle_empty_frame(self, current_time_sec: float) -> None:
        cfg = self.config
        ard = self._arduino
        s = self._state
        arduino_send_fps = cfg["arduino_send_fps"]
        send_interval = 1.0 / arduino_send_fps if arduino_send_fps > 0 else 0
        if (current_time_sec - s.last_send_time) < send_interval:
            return
        if not ard.ready:
            return

        self.empty_streak += 1
        threshold = cfg["empty_captures_before_idle"]

        if self.empty_streak < threshold:
            ard.drop_current_buffer()
        elif self.idle_images:
            path = self.idle_images[self.idle_index % len(self.idle_images)]
            binary = preprocess_idle_image(path, cfg)
            if binary is not None:
                ard.send_frame(binary, cassette=self._random_cassette())
                print(f"[Idle] Sent {os.path.basename(path)}")
            self.idle_index += 1
        else:
            ard.drop_current_buffer()

        s.last_send_time = current_time_sec
