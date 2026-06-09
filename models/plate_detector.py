"""
models/plate_detector.py
YOLOv8 license plate detector.
Operates on vehicle-cropped ROIs returned by VehicleDetector.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("numplate.plate_detector")


@dataclass
class PlateDetection:
    """Single plate detection result, relative to the vehicle crop."""
    bbox: np.ndarray    # [x1, y1, x2, y2] relative to the crop passed in
    confidence: float
    # Absolute coords in the original frame (set by pipeline after crop offset)
    abs_bbox: Optional[np.ndarray] = None


class PlateDetector:
    """
    YOLOv8 license plate detector.

    If the fine-tuned weights file is not found, falls back to the
    generic COCO model (less accurate but still functional for demos).

    Parameters
    ----------
    weights : str
        Path to fine-tuned plate detector weights.
    fallback_weights : str
        Fallback model name/path if `weights` file does not exist.
    confidence : float
        Minimum detection confidence.
    iou_threshold : float
        NMS IoU threshold.
    device : str
        Inference device.
    imgsz : int
        Inference image size (smaller = faster for crop-sized inputs).
    """

    def __init__(
        self,
        weights: str = "models/weights/plate_detector-3/weights/plate_detector_best.pt",
        fallback_weights: str = "yolov8n.pt",
        confidence: float = 0.40,
        iou_threshold: float = 0.45,
        device: str = "auto",
        imgsz: int = 320,
    ) -> None:
        self._confidence = confidence
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._device = self._resolve_device(device)

        actual_weights = weights if os.path.exists(weights) else fallback_weights
        if actual_weights == fallback_weights:
            logger.warning(
                "Fine-tuned plate weights not found at '%s'. "
                "Falling back to '%s' — run scripts/train_plate_detector.py to "
                "improve accuracy.",
                weights,
                fallback_weights,
            )

        logger.info(
            "Loading plate detector — weights=%s  device=%s", actual_weights, self._device
        )

        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(actual_weights)
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        logger.info("Plate detector ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        crop: np.ndarray,
        vehicle_offset: Optional[Tuple[int, int]] = None,
    ) -> List[PlateDetection]:
        """
        Detect license plates within a vehicle crop.

        Parameters
        ----------
        crop : np.ndarray
            BGR image crop of a single vehicle.
        vehicle_offset : (x1, y1), optional
            Top-left pixel offset of the crop in the original frame.
            When provided, ``abs_bbox`` on each result is populated.

        Returns
        -------
        List of PlateDetection sorted by confidence (highest first).
        """
        if crop is None or crop.size == 0:
            return []

        results = self._model.predict(
            crop,
            conf=self._confidence,
            iou=self._iou,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )

        detections: List[PlateDetection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()

            abs_bbox: Optional[np.ndarray] = None
            if vehicle_offset is not None:
                ox, oy = vehicle_offset
                abs_bbox = xyxy + np.array([ox, oy, ox, oy], dtype=np.float32)

            detections.append(
                PlateDetection(bbox=xyxy, confidence=conf, abs_bbox=abs_bbox)
            )
            logger.debug("Plate detected  conf=%.2f  bbox=%s", conf, xyxy.tolist())

        # Return highest-confidence first
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def best(
        self,
        crop: np.ndarray,
        vehicle_offset: Optional[Tuple[int, int]] = None,
    ) -> Optional[PlateDetection]:
        """Convenience: return only the highest-confidence plate (or None)."""
        results = self.detect(crop, vehicle_offset=vehicle_offset)
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
