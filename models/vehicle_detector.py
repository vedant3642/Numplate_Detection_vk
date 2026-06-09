"""
models/vehicle_detector.py
YOLOv8 vehicle detection wrapper.
Filters COCO detections to: car, truck, bus, motorcycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("numplate.vehicle_detector")

# COCO class IDs relevant to vehicles
_DEFAULT_VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck


@dataclass
class VehicleDetection:
    """Single vehicle detection result."""
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixel coords
    class_id: int
    class_name: str
    confidence: float


class VehicleDetector:
    """
    YOLOv8-based vehicle detector.

    Parameters
    ----------
    weights : str
        Path to model weights or Ultralytics model name (e.g. 'yolov8n.pt').
    confidence : float
        Minimum detection confidence (0–1).
    iou_threshold : float
        NMS IoU threshold.
    device : str
        'cpu', 'cuda', 'mps', or 'auto'.
    target_classes : set[int]
        COCO class IDs to retain.
    imgsz : int
        Inference image size.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.45,
        iou_threshold: float = 0.50,
        device: str = "auto",
        target_classes: Optional[List[int]] = None,
        imgsz: int = 640,
    ) -> None:
        self._confidence = confidence
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._target_classes: set[int] = (
            set(target_classes) if target_classes else _DEFAULT_VEHICLE_CLASSES
        )

        # Resolve device
        self._device = self._resolve_device(device)

        logger.info(
            "Loading vehicle detector — weights=%s  device=%s", weights, self._device
        )
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(weights)
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        # Warm-up pass to pre-allocate GPU memory
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            device=self._device,
            verbose=False,
            conf=self._confidence,
        )
        logger.info("Vehicle detector ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """
        Run detection on a single BGR frame.

        Returns a list of VehicleDetection objects filtered to vehicle classes.
        """
        results = self._model.predict(
            frame,
            conf=self._confidence,
            iou=self._iou,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )

        detections: List[VehicleDetection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self._target_classes:
                continue

            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

            class_name = (
                result.names.get(cls_id, str(cls_id))
                if result.names
                else str(cls_id)
            )

            detections.append(
                VehicleDetection(
                    bbox=xyxy,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                )
            )
            logger.debug(
                "Vehicle detected: %s  conf=%.2f  bbox=%s",
                class_name,
                conf,
                xyxy.tolist(),
            )

        return detections

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
