"""
tracking/tracker.py
DeepSORT wrapper that integrates with the pipeline's vehicle detections.
Returns (track_id, bbox) pairs and exposes lost track IDs for flushing
the TrackBuffer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("numplate.tracker")


@dataclass
class TrackedVehicle:
    """A confirmed, active vehicle track."""
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixel coords
    class_name: str = "vehicle"
    confidence: float = 1.0


class DeepSORTTracker:
    """
    DeepSORT-based multi-object tracker.

    Parameters
    ----------
    max_age : int
        Maximum frames a track survives without a matching detection.
    n_init : int
        Number of consecutive detections required to confirm a track.
    max_cosine_distance : float
        Re-ID feature distance threshold.
    nn_budget : int
        Maximum number of appearance descriptors stored per track.
    embedder : str | None
        Feature extractor for appearance — None → uses default MobileNetV2.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
        embedder: Optional[str] = None,
    ) -> None:
        self._max_age = max_age
        self._active_ids: Set[int] = set()
        self._lost_ids: Set[int] = set()

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore

            # Auto-detect GPU availability for the appearance embedder
            try:
                import torch
                _embedder_gpu = torch.cuda.is_available()
            except ImportError:
                _embedder_gpu = False

            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_cosine_distance=max_cosine_distance,
                nn_budget=nn_budget,
                embedder=embedder or "mobilenet",
                half=False,
                bgr=True,       # Our frames are OpenCV BGR
                embedder_gpu=_embedder_gpu,
            )
            logger.info(
                "DeepSORT initialised — max_age=%d  n_init=%d", max_age, n_init
            )
        except ImportError as exc:
            raise ImportError(
                "deep_sort_realtime is not installed.\n"
                "Run: pip install deep-sort-realtime"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list,   # List of VehicleDetection from VehicleDetector
        frame: np.ndarray,
    ) -> Tuple[List[TrackedVehicle], Set[int]]:
        """
        Update tracker with new detections.

        Parameters
        ----------
        detections : list[VehicleDetection]
            Raw detections for this frame.
        frame : np.ndarray
            Full BGR frame (used by DeepSORT for appearance embedding).

        Returns
        -------
        (active_tracks, lost_ids)
            active_tracks — confirmed tracks this frame.
            lost_ids      — track IDs that just became inactive.
        """
        # Convert VehicleDetection → DeepSORT input format:
        # [[left, top, width, height], confidence, class_name]
        ds_input = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w, h = x2 - x1, y2 - y1
            ds_input.append(([x1, y1, w, h], det.confidence, det.class_name))

        raw_tracks = self._tracker.update_tracks(ds_input, frame=frame)

        active_tracks: List[TrackedVehicle] = []
        current_ids: Set[int] = set()

        for track in raw_tracks:
            if not track.is_confirmed():
                continue
            tid = int(track.track_id)
            ltrb = track.to_ltrb()           # [x1, y1, x2, y2]
            current_ids.add(tid)

            active_tracks.append(
                TrackedVehicle(
                    track_id=tid,
                    bbox=np.array(ltrb, dtype=np.float32),
                    class_name=track.get_det_class() or "vehicle",
                    confidence=track.get_det_conf() or 1.0,
                )
            )

        # Detect IDs that were active last frame but are gone now → lost
        self._lost_ids = self._active_ids - current_ids
        self._active_ids = current_ids

        if self._lost_ids:
            logger.debug("Lost tracks this frame: %s", self._lost_ids)

        return active_tracks, self._lost_ids

    def get_lost_ids(self) -> Set[int]:
        """Return the set of track IDs lost in the last update call."""
        return self._lost_ids

    def reset(self) -> None:
        """Clear internal state (call between video files)."""
        self._active_ids.clear()
        self._lost_ids.clear()
        try:
            self._tracker.delete_all_tracks()
        except AttributeError:
            pass
