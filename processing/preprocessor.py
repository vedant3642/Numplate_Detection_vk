"""
processing/preprocessor.py
Image preprocessing steps applied to plate crops before OCR.

Indian-specific optimisations:
- CLAHE to handle yellow/white background glare
- Rightmost-85% crop to remove the IND emblem on the left side
- Minimum height upscaling so OCR engines see readable characters
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger("numplate.preprocessor")


class Preprocessor:
    """
    Stateless image preprocessor for license plate crops.

    Parameters
    ----------
    clahe_enabled : bool
        Apply CLAHE contrast enhancement.
    clip_limit : float
        CLAHE clip limit (higher → more contrast).
    tile_grid_size : tuple[int, int]
        CLAHE tile grid size.
    plate_crop_ratio : float
        Fraction of the plate width to keep (from the right).
        0.85 trims the left 15%, removing the IND emblem on Indian plates.
    min_plate_height : int
        Upscale the crop if its height is below this value.
    min_plate_width : int
        Upscale the crop if its width is below this value.
    """

    def __init__(
        self,
        clahe_enabled: bool = True,
        clip_limit: float = 3.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        plate_crop_ratio: float = 0.85,
        min_plate_height: int = 64,
        min_plate_width: int = 100,
    ) -> None:
        self._clahe_enabled = clahe_enabled
        self._plate_crop_ratio = plate_crop_ratio
        self._min_plate_height = min_plate_height
        self._min_plate_width = min_plate_width

        if clahe_enabled:
            self._clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=tile_grid_size,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Full preprocessing chain:
          1. Trim IND emblem (left side crop)
          2. Upscale if too small
          3. CLAHE on L-channel (LAB colour space)

        Parameters
        ----------
        plate_img : np.ndarray
            BGR crop of the detected plate region.

        Returns
        -------
        Preprocessed BGR image ready for OCR.
        """
        if plate_img is None or plate_img.size == 0:
            logger.warning("Preprocessor received empty image — skipping.")
            return plate_img

        img = self._trim_ind_emblem(plate_img)
        img = self._upscale_if_needed(img)

        if self._clahe_enabled:
            img = self.apply_clahe(img)

        return img

    def apply_clahe(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L channel of the image (LAB colour space).
        Returns a BGR image.
        """
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trim_ind_emblem(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the rightmost `plate_crop_ratio` fraction of the plate.
        This removes the blue IND symbol on Indian plates.
        """
        h, w = img.shape[:2]
        x_start = int(w * (1.0 - self._plate_crop_ratio))
        cropped = img[:, x_start:]
        logger.debug(
            "Trimmed IND emblem: original_w=%d  new_w=%d", w, cropped.shape[1]
        )
        return cropped

    def _upscale_if_needed(self, img: np.ndarray) -> np.ndarray:
        """Upscale the image if it is smaller than the minimum dimensions."""
        h, w = img.shape[:2]
        if h >= self._min_plate_height and w >= self._min_plate_width:
            return img

        scale_h = self._min_plate_height / h if h < self._min_plate_height else 1.0
        scale_w = self._min_plate_width / w if w < self._min_plate_width else 1.0
        scale = max(scale_h, scale_w)

        new_w = max(int(w * scale), self._min_plate_width)
        new_h = max(int(h * scale), self._min_plate_height)

        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug(
            "Upscaled plate: (%d, %d) → (%d, %d)", w, h, new_w, new_h
        )
        return upscaled
