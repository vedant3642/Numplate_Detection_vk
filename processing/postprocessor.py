"""
processing/postprocessor.py
Post-OCR logic:
  - Regex validation of plate format
  - TrackBuffer: accumulates multi-frame OCR reads per track ID
  - Deduplication strategies: majority vote OR best-confidence
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("numplate.postprocessor")

# Indian plate pattern (default)
_DEFAULT_REGEX = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{1,4}$"


 
# Regex validator
 

class PlateValidator:
    """
    Validates raw OCR text against a country-specific plate regex.

    Parameters
    ----------
    pattern : str
        Regular expression that a valid plate must fully match.
    """

    # Common OCR confusions: letter ↔ digit
    _LETTER_TO_DIGIT = str.maketrans("OISBG", "01589")
    _DIGIT_TO_LETTER = str.maketrans("015", "OIS")

    def __init__(self, pattern: str = _DEFAULT_REGEX) -> None:
        self._re = re.compile(pattern)

    def is_valid(self, text: str) -> bool:
        """Return True if the text matches the plate pattern."""
        clean = text.upper().strip().replace(" ", "").replace("-", "")
        result = bool(self._re.fullmatch(clean))
        if not result:
            logger.debug("Regex rejected: '%s'", text)
        return result

    def clean(self, text: str) -> str:
        """Normalise text to uppercase with no spaces or dashes."""
        return text.upper().strip().replace(" ", "").replace("-", "")

    def correct_ocr_chars(self, text: str) -> str:
        """
        Fix common OCR letter↔digit confusions for Indian plates.

        Indian format: [A-Z]{2} [0-9]{1,2} [A-Z]{1,3} [0-9]{1,4}
        We know which positions MUST be letters and which MUST be digits,
        so we can fix O→0, I→1, S→5 in digit positions and vice versa.
        """
        clean = self.clean(text)
        if len(clean) < 6:
            return clean  # too short to reason about

        # Positions 0-1: state code → MUST be letters
        part = list(clean)
        for i in range(min(2, len(part))):
            if part[i].isdigit():
                part[i] = str(part[i]).translate(self._DIGIT_TO_LETTER)

        # Find where the first digit group starts (positions 2+)
        # Positions 2-(2+n): district code → MUST be digits (1-2 digits)
        i = 2
        while i < len(part) and i < 4 and (part[i].isdigit() or part[i] in 'OISBG'):
            if part[i].isalpha():
                part[i] = part[i].translate(self._LETTER_TO_DIGIT)
            i += 1
        district_end = i

        # Next 1-3 chars: series letters → MUST be letters
        while i < len(part) and i < district_end + 3 and (part[i].isalpha() or part[i] in '015'):
            if part[i].isdigit():
                part[i] = str(part[i]).translate(self._DIGIT_TO_LETTER)
            i += 1

        # Remaining chars: registration number → MUST be digits
        while i < len(part):
            if part[i].isalpha():
                part[i] = part[i].translate(self._LETTER_TO_DIGIT)
            i += 1

        corrected = "".join(part)
        if corrected != clean:
            logger.info("OCR char correction: '%s' → '%s'", clean, corrected)
        return corrected


 
# Per-track OCR read record
 

@dataclass
class OCRRead:
    """A single OCR read for one track."""
    text: str
    confidence: float
    frame_no: int


# Flushed track result 

@dataclass
class CommittedDetection:
    """Final de-duplicated result for a track that has ended."""
    track_id: int
    plate_number: str
    confidence: float
    first_frame: int
    total_reads: int


 
# Track buffer
 

class TrackBuffer:
    """
    Accumulates OCR reads per track_id across multiple frames.
    When a track is lost, flush() collapses them into a single best read.

    Parameters
    ----------
    validator : PlateValidator
        Used to discard invalid reads before buffering.
    strategy : str
        'majority_vote' — pick the most-frequently-occurring plate text.
        'best_confidence' — pick the read with the highest confidence.
    min_reads : int
        Minimum number of valid reads needed before a flush is accepted.
        Prevents garbage from single-frame tracks.
    """

    def __init__(
        self,
        validator: PlateValidator,
        strategy: str = "majority_vote",
        min_reads: int = 3,
    ) -> None:
        self._validator = validator
        self._strategy = strategy
        self._min_reads = min_reads

        # track_id → list of valid reads
        self._buffer: Dict[int, List[OCRRead]] = defaultdict(list)
        # track_id → first frame number seen
        self._first_frame: Dict[int, int] = {}


    # Public API  

    def add(
        self,
        track_id: int,
        text: str,
        confidence: float,
        frame_no: int,
    ) -> bool:
        """
        Buffer an OCR read for a track, if it passes regex validation.
        Applies OCR character correction as a fallback if raw text fails.

        Returns True if the read was accepted, False if rejected.
        """
        clean = self._validator.clean(text)

        # Try raw text first, then corrected text
        if self._validator.is_valid(clean):
            accepted_text = clean
        else:
            corrected = self._validator.correct_ocr_chars(clean)
            if self._validator.is_valid(corrected):
                accepted_text = corrected
            else:
                return False

        if track_id not in self._first_frame:
            self._first_frame[track_id] = frame_no

        self._buffer[track_id].append(
            OCRRead(text=accepted_text, confidence=confidence, frame_no=frame_no)
        )
        logger.debug(
            "TrackBuffer.add  track=%d  text='%s'  conf=%.2f  total_reads=%d",
            track_id,
            accepted_text,
            confidence,
            len(self._buffer[track_id]),
        )
        return True

    def flush(self, track_id: int) -> Optional[CommittedDetection]:
        """
        Collapse all buffered reads for a lost track into one result.

        Returns None if there are fewer valid reads than min_reads.
        """
        reads = self._buffer.pop(track_id, [])
        first_frame = self._first_frame.pop(track_id, 0)

        if len(reads) < self._min_reads:
            logger.debug(
                "TrackBuffer.flush  track=%d  DROPPED (only %d reads < min %d)",
                track_id,
                len(reads),
                self._min_reads,
            )
            return None

        if self._strategy == "best_confidence":
            best = max(reads, key=lambda r: r.confidence)
            plate_text = best.text
            confidence = best.confidence
        else:  # majority_vote (default)
            counter = Counter(r.text for r in reads)
            plate_text = counter.most_common(1)[0][0]
            # Average confidence of reads that match the winning text
            matching = [r for r in reads if r.text == plate_text]
            confidence = sum(r.confidence for r in matching) / len(matching)

        logger.info(
            "TrackBuffer.flush  track=%d  plate='%s'  conf=%.2f  reads=%d",
            track_id,
            plate_text,
            confidence,
            len(reads),
        )

        return CommittedDetection(
            track_id=track_id,
            plate_number=plate_text,
            confidence=round(confidence, 4),
            first_frame=first_frame,
            total_reads=len(reads),
        )

    def flush_all(self) -> List[CommittedDetection]:
        """Flush every buffered track (call at end-of-video)."""
        all_ids = list(self._buffer.keys())
        results = []
        for tid in all_ids:
            result = self.flush(tid)
            if result:
                results.append(result)
        return results

    def active_tracks(self) -> Set[int]:
        """Return set of track IDs currently buffered."""
        return set(self._buffer.keys())

    def read_count(self, track_id: int) -> int:
        """Return how many valid reads are buffered for a track."""
        return len(self._buffer.get(track_id, []))
