"""
storage/models.py
SQLAlchemy ORM model for persisted plate detections.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Detection(Base):
    """
    One row = one unique license plate seen in the video.

    Columns
    -------
    plate_number    Normalised plate string (PK — one row per unique plate).
    first_seen_time Timestamp of the first frame this plate appeared.
    track_id        DeepSORT track ID at first detection.
    confidence      Average OCR confidence for the winning plate text.
    frame_no        Frame number of first appearance.
    vehicle_class   COCO vehicle class name (car, truck, bus, motorcycle).
    total_reads     Number of valid OCR reads accumulated for this track.
    """

    __tablename__ = "detections"

    plate_number: str = Column(String(20), primary_key=True, nullable=False)
    first_seen_time: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    track_id: int = Column(Integer, nullable=False)
    confidence: float = Column(Float, nullable=False)
    frame_no: int = Column(Integer, nullable=False)
    vehicle_class: str = Column(String(32), nullable=True)
    total_reads: int = Column(Integer, default=1)

    def __repr__(self) -> str:
        return (
            f"<Detection plate={self.plate_number!r} "
            f"track={self.track_id} conf={self.confidence:.2f}>"
        )
