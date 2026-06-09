"""
pipeline.py
Main orchestration loop — ties every module together.

Flow per frame:
  1. Read frame from VideoCapture
  2. Vehicle detection (YOLOv8 COCO)
  3. DeepSORT tracking → active tracks + lost track IDs
  4. For each tracked vehicle:
       a. Crop vehicle ROI
       b. Plate detection within crop
       c. Preprocess plate crop (CLAHE, trim emblem, upscale)
       d. OCR → (text, confidence)
       e. Add to TrackBuffer if regex-valid
  5. Flush lost tracks → upsert to DB
  6. Annotate and display frame
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import cv2
import numpy as np

from models.ocr_engine import OCREngine
from models.plate_detector import PlateDetector
from models.vehicle_detector import VehicleDetector
from processing.postprocessor import CommittedDetection, TrackBuffer
from processing.preprocessor import Preprocessor
from storage.database import DatabaseManager
from tracking.tracker import DeepSORTTracker
from utils.logger import get_logger
from utils.visualizer import Visualizer

logger = get_logger("numplate.pipeline")


@dataclass
class PipelineStats:
    frames_processed: int = 0
    vehicles_tracked: int = 0
    plates_read: int = 0
    plates_committed: int = 0
    start_time: float = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def fps(self) -> float:
        return self.frames_processed / self.elapsed if self.elapsed > 0 else 0.0


class NumPlatePipeline:
    """
    End-to-end number plate detection pipeline.

    Parameters
    ----------
    vehicle_detector : VehicleDetector
    plate_detector   : PlateDetector
    ocr_engine       : OCREngine
    tracker          : DeepSORTTracker
    preprocessor     : Preprocessor
    track_buffer     : TrackBuffer
    db               : DatabaseManager
    visualizer       : Visualizer | None
    show_window      : bool         Show OpenCV window.
    skip_frames      : int          Process every Nth frame (1 = all).
    save_video       : bool         Write annotated output video.
    output_video_path: str          Path for output video file.
    """

    def __init__(
        self,
        vehicle_detector: VehicleDetector,
        plate_detector: PlateDetector,
        ocr_engine: OCREngine,
        tracker: DeepSORTTracker,
        preprocessor: Preprocessor,
        track_buffer: TrackBuffer,
        db: DatabaseManager,
        visualizer: Optional[Visualizer] = None,
        show_window: bool = True,
        skip_frames: int = 1,
        save_video: bool = False,
        output_video_path: str = "output/result.mp4",
    ) -> None:
        self._vd = vehicle_detector
        self._pd = plate_detector
        self._ocr = ocr_engine
        self._tracker = tracker
        self._pre = preprocessor
        self._buf = track_buffer
        self._db = db
        self._vis = visualizer
        self._show = show_window
        self._skip = max(1, skip_frames)
        self._save_video = save_video
        self._output_path = output_video_path

        # Map track_id → most recent plate text (for display continuity)
        self._plate_labels: Dict[int, str] = {}
        self._plate_confs: Dict[int, float] = {}
        self._vehicle_classes: Dict[int, str] = {}

        self._stats = PipelineStats()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, source: str | int) -> PipelineStats:
        """
        Process a video source until completion or until the user presses 'q'.

        Parameters
        ----------
        source : str | int
            Path to a video file, an RTSP URL, or 0 for webcam.

        Returns
        -------
        PipelineStats with summary of the run.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Video opened — source=%r  %dx%d  %.1f fps  ~%d frames",
            source, width, height, fps_native, total,
        )

        # Set up display window properly so it refreshes without needing focus
        if self._show:
            cv2.namedWindow("NumPlate Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("NumPlate Detection", min(width, 1280), min(height, 720))

        writer: Optional[cv2.VideoWriter] = None
        if self._save_video:
            import os; os.makedirs(
                os.path.dirname(self._output_path) or ".", exist_ok=True
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self._output_path, fourcc, fps_native, (width, height)
            )

        self._stats = PipelineStats(start_time=time.monotonic())
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream.")
                    break

                frame_idx += 1

                # Process every Nth frame; still annotate skipped frames
                if frame_idx % self._skip != 0:
                    if self._show:
                        cv2.imshow("NumPlate Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                self._stats.frames_processed += 1
                annotated = self._process_frame(frame, frame_idx)

                if writer:
                    writer.write(annotated)

                if self._show:
                    cv2.imshow("NumPlate Detection", annotated)
                    # waitKey(30) ensures the window repaints on Windows
                    # without requiring user focus/click
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        logger.info("User pressed 'q' — stopping.")
                        break

        finally:
            # Flush any remaining tracks at end-of-video
            remaining = self._buf.flush_all()
            for det in remaining:
                self._commit(det, frame_idx, fps_native)

            cap.release()
            if writer:
                writer.release()
            if self._show:
                cv2.destroyAllWindows()
            self._db.close()

        self._print_summary()
        return self._stats

    # ------------------------------------------------------------------
    # Per-frame logic
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        annotated = frame.copy()

        # --- Step 1: Detect vehicles ---
        vehicle_dets = self._vd.detect(frame)

        # --- Step 2: Track ---
        active_tracks, lost_ids = self._tracker.update(vehicle_dets, frame)
        self._stats.vehicles_tracked = max(
            self._stats.vehicles_tracked, len(active_tracks)
        )

        # --- Step 3: Flush lost tracks → commit detections ---
        for tid in lost_ids:
            committed = self._buf.flush(tid)
            if committed:
                fps_native = 30.0  # approximate; override in run() if needed
                self._commit(committed, frame_idx, fps_native)
            self._plate_labels.pop(tid, None)
            self._plate_confs.pop(tid, None)
            self._vehicle_classes.pop(tid, None)

        # --- Step 4: Per-track plate detection + OCR ---
        for track in active_tracks:
            tid = track.track_id
            self._vehicle_classes[tid] = track.class_name
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            plate_det = self._pd.best(
                vehicle_crop, vehicle_offset=(x1, y1)
            )

            plate_text_display = self._plate_labels.get(tid, "")
            plate_conf_display = self._plate_confs.get(tid, 0.0)
            abs_plate_bbox = None

            if plate_det is not None:
                # Crop plate from vehicle crop
                px1, py1, px2, py2 = (int(v) for v in plate_det.bbox)
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = (
                    min(vehicle_crop.shape[1] - 1, px2),
                    min(vehicle_crop.shape[0] - 1, py2),
                )
                plate_crop = vehicle_crop[py1:py2, px1:px2]

                if plate_crop.size > 0:
                    # Preprocess
                    plate_crop = self._pre.process(plate_crop)

                    # OCR
                    text, conf = self._ocr.read(plate_crop)
                    self._stats.plates_read += 1

                    if text:
                        self._buf.add(tid, text, conf, frame_idx)
                        self._plate_labels[tid] = text
                        self._plate_confs[tid] = conf
                        plate_text_display = text
                        plate_conf_display = conf

                if plate_det.abs_bbox is not None:
                    abs_plate_bbox = plate_det.abs_bbox

            # --- Annotate ---
            if self._vis:
                self._vis.draw_vehicle_box(
                    annotated,
                    track.bbox,
                    tid,
                    plate_text=plate_text_display,
                    vehicle_class=track.class_name,
                    confidence=plate_conf_display,
                )
                if abs_plate_bbox is not None:
                    self._vis.draw_plate_box(
                        annotated,
                        abs_plate_bbox,
                        text=plate_text_display,
                        confidence=plate_conf_display,
                    )

        # --- HUD ---
        if self._vis:
            self._vis.draw_stats_overlay(
                annotated,
                vehicle_count=self._db.get_vehicle_count(),
                frame_no=frame_idx,
            )

        return annotated

    # ------------------------------------------------------------------
    # Commit helper
    # ------------------------------------------------------------------

    def _commit(
        self,
        det: CommittedDetection,
        frame_idx: int,
        fps: float,
    ) -> None:
        """Persist a committed detection to the database."""
        # Compute timestamp from frame index
        seconds = frame_idx / max(fps, 1.0)
        ts = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(seconds=seconds)

        vclass = self._vehicle_classes.get(det.track_id, "vehicle")
        inserted = self._db.upsert_detection(
            plate_number=det.plate_number,
            track_id=det.track_id,
            confidence=det.confidence,
            frame_no=det.first_frame,
            first_seen_time=ts,
            total_reads=det.total_reads,
            vehicle_class=vclass,
        )
        if inserted:
            self._stats.plates_committed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        logger.info("=" * 55)
        logger.info("Pipeline finished in %.1f s", self._stats.elapsed)
        logger.info("  Frames processed : %d", self._stats.frames_processed)
        logger.info("  Avg FPS          : %.1f", self._stats.fps)
        logger.info("  Plates committed : %d", self._stats.plates_committed)
        logger.info(
            "  Unique vehicles  : %d", self._db.get_vehicle_count()
        )
        logger.info("=" * 55)
