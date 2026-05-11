"""
Display module for Water Drawing App.

Owns the OpenCV window, overlay toggle, FPS counter, and per-frame render.
No knowledge of Arduino or streaming policy — render() takes a snapshot
of the streaming state and a FrameMetrics bundle.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from streaming import FrameMetrics, StreamingState


@dataclass
class FpsState:
    frame_count: int
    fps: float
    fps_update_time: float


class Display:
    """OpenCV window owner: render, overlay toggle, fullscreen, FPS."""

    def __init__(self, config: dict, window_name: str = "Water Drawing"):
        self.config = config
        self.window_name = window_name
        self.overlay_visible = False
        self.fps_state = FpsState(
            frame_count=0, fps=0.0, fps_update_time=cv2.getTickCount()
        )
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if config["fullscreen"]:
            _set_fullscreen(self.window_name, True)

    def render(
        self,
        frame: np.ndarray,
        metrics: FrameMetrics,
        streaming_state: StreamingState,
        reference_bg_set: bool,
    ) -> None:
        cfg = self.config
        display_original = frame
        if cfg["flip_vertical"]:
            display_original = cv2.flip(display_original, 0)
        if cfg["flip_horizontal"]:
            display_original = cv2.flip(display_original, 1)
        display_frame = _create_stacked_display(
            display_original, metrics.binary, cfg["display_scale"]
        )
        if self.overlay_visible:
            status = self._format_status(streaming_state, metrics)
            display_frame = _draw_overlay(
                display_frame,
                cfg,
                self.fps_state.fps,
                metrics.white_ratio,
                has_reference_bg=reference_bg_set,
                stationary_status=status,
            )
        if cfg["fullscreen"]:
            display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow(self.window_name, display_frame)

    def update_fps(self) -> None:
        s = self.fps_state
        count = s.frame_count + 1
        current_time = cv2.getTickCount()
        elapsed = (current_time - s.fps_update_time) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            self.fps_state = FpsState(
                frame_count=0, fps=count / elapsed, fps_update_time=current_time
            )
        else:
            self.fps_state = FpsState(
                frame_count=count, fps=s.fps, fps_update_time=s.fps_update_time
            )

    def toggle_overlay(self) -> None:
        self.overlay_visible = not self.overlay_visible
        print(f"Overlay: {'ON' if self.overlay_visible else 'OFF'}")

    def toggle_fullscreen(self) -> None:
        cfg = self.config
        cfg["fullscreen"] = not cfg["fullscreen"]
        _set_fullscreen(self.window_name, cfg["fullscreen"])
        print(f"Fullscreen: {'ON' if cfg['fullscreen'] else 'OFF'}")

    def close(self) -> None:
        cv2.destroyAllWindows()

    def _format_status(
        self, s: StreamingState, metrics: FrameMetrics
    ) -> str:
        cfg = self.config
        stationary_threshold = cfg["stationary_threshold"]
        require_stationary = cfg["require_stationary_for_send"]
        arduino_send_fps = cfg["arduino_send_fps"]
        send_interval = 1.0 / arduino_send_fps if arduino_send_fps > 0 else 0
        pixel_change_ratio = metrics.pixel_change_ratio

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


def _create_stacked_display(
    original: np.ndarray, binary: np.ndarray, scale: int
) -> np.ndarray:
    bin_h, bin_w = binary.shape[:2]
    binary_display_size = (bin_w * scale, bin_h * scale)

    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary_scaled = cv2.resize(
        binary_bgr, binary_display_size, interpolation=cv2.INTER_NEAREST
    )

    orig_h, orig_w = original.shape[:2]
    target_width = binary_display_size[0]
    aspect_ratio = orig_h / orig_w
    target_height = int(target_width * aspect_ratio)

    original_scaled = cv2.resize(
        original, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )

    return cv2.vconcat([original_scaled, binary_scaled])


def _draw_overlay(
    frame: np.ndarray,
    config: dict,
    fps: float,
    white_ratio: float,
    has_reference_bg: bool = False,
    stationary_status: str = "",
) -> np.ndarray:
    display = frame.copy()
    lines = [
        f"Diff Threshold: {config['difference_threshold']} (+/- to adjust)",
        f"Diff Mode: {config['diff_mode']} (m=cycle)",
        f"Morph: erode={config['morph_erode']} dilate={config['morph_dilate']} (e/E, l/L)",
        f"FPS: {fps:.1f}  White: {white_ratio * 100:.1f}%",
        f"Ref BG: {'SET' if has_reference_bg else 'NOT SET'} (r to capture)",
        f"Stationary delay: {config['stationary_delay_ms']}ms  {stationary_status}",
        f"Fullscreen: {'ON' if config['fullscreen'] else 'OFF'} (f)",
        "s=save, SPACE=send, q=quit",
    ]
    y = 25
    for line in lines:
        cv2.putText(
            display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        y += 20
    return display


def _set_fullscreen(window_name: str, enabled: bool) -> None:
    if enabled:
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
    else:
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
        )
