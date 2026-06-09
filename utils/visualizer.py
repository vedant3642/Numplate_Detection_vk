"""
utils/visualizer.py
Frame annotation utilities: vehicle bounding boxes, plate highlights,
track ID labels, and the HUD stats overlay.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("numplate.visualizer")

# Colour palette (BGR)
_VEHICLE_BOX_COLOR = (128, 255, 0)    # bright green
_PLATE_BOX_COLOR   = (0, 165, 255)    # orange
_TEXT_BG_COLOR     = (20, 20, 20)     # near-black
_TEXT_COLOR        = (255, 255, 255)  # white
_HUD_BG_COLOR      = (15, 15, 15, 180)  # semi-transparent dark


class Visualizer:
    """
    Draws detection and tracking annotations on video frames.

    Parameters
    ----------
    vehicle_box_color : tuple BGR colour for vehicle boxes.
    plate_box_color   : tuple BGR colour for plate boxes.
    text_color        : tuple BGR colour for text labels.
    font_scale        : float OpenCV font scale.
    line_thickness    : int   Box line thickness.
    """

    def __init__(
        self,
        vehicle_box_color: Tuple[int, int, int] = _VEHICLE_BOX_COLOR,
        plate_box_color: Tuple[int, int, int] = _PLATE_BOX_COLOR,
        text_color: Tuple[int, int, int] = _TEXT_COLOR,
        font_scale: float = 0.65,
        line_thickness: int = 2,
    ) -> None:
        self._veh_color = tuple(vehicle_box_color)
        self._plt_color = tuple(plate_box_color)
        self._txt_color = tuple(text_color)
        self._font_scale = font_scale
        self._thickness = line_thickness
        self._font = cv2.FONT_HERSHEY_DUPLEX

        # FPS calculation
        self._fps_times: list = []
        self._fps_window = 30  # rolling window in frames

    # ------------------------------------------------------------------
    # Public drawing methods
    # ------------------------------------------------------------------

    def draw_vehicle_box(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        track_id: int,
        plate_text: str = "",
        vehicle_class: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Draw vehicle bounding box with track ID and plate text label."""
        x1, y1, x2, y2 = (int(v) for v in bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), self._veh_color, self._thickness)

        label_parts = [f"#{track_id}"]
        if vehicle_class:
            label_parts.append(vehicle_class)
        if plate_text:
            label_parts.append(plate_text)
        if confidence > 0:
            label_parts.append(f"{confidence:.0%}")

        label = "  ".join(label_parts)
        self._draw_label(frame, label, (x1, y1 - 6), self._veh_color)

    def draw_plate_box(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        text: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Draw highlighted plate bounding box with OCR text overlay."""
        x1, y1, x2, y2 = (int(v) for v in bbox)

        # Thicker box for plates
        cv2.rectangle(frame, (x1, y1), (x2, y2), self._plt_color, self._thickness + 1)

        if text:
            label = f"{text}  {confidence:.0%}" if confidence > 0 else text
            self._draw_label(frame, label, (x1, y2 + 18), self._plt_color)

    def draw_stats_overlay(
        self,
        frame: np.ndarray,
        vehicle_count: int,
        fps: Optional[float] = None,
        frame_no: int = 0,
    ) -> None:
        """Draw the HUD stats box in the top-left corner."""
        self._update_fps()
        display_fps = fps if fps is not None else self._get_fps()

        h, w = frame.shape[:2]

        # Semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (300, 105), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Lines of text
        lines = [
            f"Vehicles : {vehicle_count}",
            f"FPS      : {display_fps:.1f}",
            f"Frame    : {frame_no}",
        ]
        y = 34
        for line in lines:
            cv2.putText(
                frame,
                line,
                (18, y),
                self._font,
                self._font_scale,
                self._txt_color,
                1,
                cv2.LINE_AA,
            )
            y += 26

    # ------------------------------------------------------------------
    # FPS helpers
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        self._fps_times.append(time.monotonic())
        if len(self._fps_times) > self._fps_window:
            self._fps_times.pop(0)

    def _get_fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        box_color: tuple,
    ) -> None:
        """Draw a filled-background text label at `origin`."""
        x, y = origin
        h, w = frame.shape[:2]

        (tw, th), baseline = cv2.getTextSize(
            text, self._font, self._font_scale, 1
        )

        # Keep label inside frame bounds
        lx1 = max(x, 0)
        ly1 = max(y - th - baseline - 4, 0)
        lx2 = min(x + tw + 6, w - 1)
        ly2 = max(y + baseline - 2, th)

        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), box_color, -1)
        cv2.putText(
            frame,
            text,
            (lx1 + 3, ly2 - baseline - 2),
            self._font,
            self._font_scale,
            self._txt_color,
            1,
            cv2.LINE_AA,
        )
