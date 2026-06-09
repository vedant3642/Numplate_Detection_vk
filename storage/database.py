"""
storage/database.py
Database session management, upsert logic, vehicle count, and CSV export.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List, Optional

import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import Session, sessionmaker

from storage.models import Base, Detection

logger = logging.getLogger("numplate.database")


class DatabaseManager:
    """
    Manages the SQLAlchemy session lifecycle and provides high-level
    persistence operations.

    Parameters
    ----------
    db_uri : str
        SQLAlchemy database URL, e.g.
        'sqlite:///detections.db' or 'postgresql://user:pass@host/db'.
    auto_create_tables : bool
        If True, create all tables on first connection.
    """

    def __init__(
        self,
        db_uri: str = "sqlite:///detections.db",
        auto_create_tables: bool = True,
    ) -> None:
        self._engine = create_engine(
            db_uri, echo=False, future=True, poolclass=NullPool
        )
        self._SessionLocal = sessionmaker(
            bind=self._engine, autoflush=False, autocommit=False
        )

        if auto_create_tables:
            Base.metadata.create_all(self._engine)
            logger.info("Database tables ensured — uri=%s", db_uri)

        self._seen_plates: set[str] = self._load_seen_plates()

    # ------------------------------------------------------------------
    # Context manager for sessions
    # ------------------------------------------------------------------

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Yield a managed SQLAlchemy session."""
        sess: Session = self._SessionLocal()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def upsert_detection(
        self,
        plate_number: str,
        track_id: int,
        confidence: float,
        frame_no: int,
        first_seen_time: Optional[datetime] = None,
        vehicle_class: str = "vehicle",
        total_reads: int = 1,
    ) -> bool:
        """
        Insert a new detection only if the plate_number has not been seen
        before (upsert-safe, primary-key collision = skip).

        Returns True if inserted, False if already existed.
        """
        if plate_number in self._seen_plates:
            logger.debug("Plate already in DB — skipping: %s", plate_number)
            return False

        detection = Detection(
            plate_number=plate_number,
            first_seen_time=first_seen_time or datetime.utcnow(),
            track_id=track_id,
            confidence=confidence,
            frame_no=frame_no,
            vehicle_class=vehicle_class,
            total_reads=total_reads,
        )

        try:
            with self.session() as sess:
                # Use merge to safely handle any race condition / retry
                sess.merge(detection)

            self._seen_plates.add(plate_number)
            logger.info(
                "NEW detection persisted — plate=%s  track=%d  conf=%.2f  frame=%d",
                plate_number,
                track_id,
                confidence,
                frame_no,
            )
            return True

        except Exception as exc:
            logger.error("Failed to persist detection %s: %s", plate_number, exc)
            return False

    def get_vehicle_count(self) -> int:
        """Return count of unique plates seen so far (in-memory, O(1))."""
        return len(self._seen_plates)

    def get_all_detections(self) -> List[dict]:
        """Fetch all detection rows from the DB as plain dicts (session-safe)."""
        with self.session() as sess:
            rows = sess.query(Detection).order_by(Detection.first_seen_time).all()
            return [
                {
                    "plate_number": r.plate_number,
                    "confidence": r.confidence,
                    "vehicle_class": r.vehicle_class,
                    "frame_no": r.frame_no,
                    "first_seen_time": r.first_seen_time,
                    "total_reads": r.total_reads,
                }
                for r in rows
            ]

    def export_to_csv(self, path: str = "output/detections.csv") -> str:
        """
        Export all detections to CSV via pandas.

        Returns the absolute path of the written file.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        detections = self.get_all_detections()
        rows = [
            {
                "plate_number": d["plate_number"],
                "first_seen_time": d["first_seen_time"],
                "confidence": d["confidence"],
                "frame_no": d["frame_no"],
                "vehicle_class": d["vehicle_class"],
                "total_reads": d["total_reads"],
            }
            for d in detections
        ]

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("Exported %d detections to '%s'", len(rows), path)
        return os.path.abspath(path)

    def close(self) -> None:
        """Dispose of the engine and close connection pools."""
        if hasattr(self, "_engine"):
            self._engine.dispose()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_seen_plates(self) -> set[str]:
        """Pre-populate the in-memory set from existing DB rows."""
        try:
            with self.session() as sess:
                rows = sess.query(Detection.plate_number).all()
            seen = {r[0] for r in rows}
            if seen:
                logger.info("Loaded %d existing plates from DB.", len(seen))
            return seen
        except Exception as exc:
            logger.warning("Could not load existing plates: %s", exc)
            return set()
