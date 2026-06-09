"""
models/ocr_engine.py
Abstract OCR interface with EasyOCR and PaddleOCR backends.
Both return (text: str, confidence: float).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("numplate.ocr")



# Abstract base

class OCREngine(ABC):
    """Base class for all OCR backends."""

    @abstractmethod
    def read(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on a preprocessed plate crop.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale plate crop.

        Returns
        -------
        (text, confidence) where text is the raw string and confidence is 0–1.
        Returns ("", 0.0) when nothing is detected.
        """

    @abstractmethod
    def read_all(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Return ALL candidate reads (not just the best), sorted by confidence."""


 
# EasyOCR backend
 

class EasyOCREngine(OCREngine):
    """
    EasyOCR-backed OCR engine.

    Parameters
    ----------
    languages : list[str]
        Language codes (e.g. ['en']).
    gpu : bool
        Whether to use GPU inference.
    allowlist : str
        Restrict recognised characters to this set.
    min_confidence : float
        Drop reads below this threshold.
    """

    def __init__(
        self,
        languages: List[str] | None = None,
        gpu: bool = True,
        allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_confidence: float = 0.45,
    ) -> None:
        self._allowlist = allowlist
        self._min_confidence = min_confidence

        try:
            import easyocr  # type: ignore

            logger.info(
                "Initialising EasyOCR — langs=%s  gpu=%s", languages or ["en"], gpu
            )
            self._reader = easyocr.Reader(
                languages or ["en"],
                gpu=gpu,
                verbose=False,
            )
            logger.info("EasyOCR ready.")
        except ImportError as exc:
            raise ImportError(
                "easyocr is not installed. Run: pip install easyocr"
            ) from exc

    # ------------------------------------------------------------------

    def read(self, image: np.ndarray) -> Tuple[str, float]:
        candidates = self.read_all(image)
        if not candidates:
            return "", 0.0
        return candidates[0]

    def read_all(self, image: np.ndarray) -> List[Tuple[str, float]]:
        if image is None or image.size == 0:
            return []

        raw = self._reader.readtext(
            image,
            allowlist=self._allowlist,
            paragraph=False,
            detail=1,
        )

        results: List[Tuple[str, float]] = []
        for (_bbox, text, conf) in raw:
            text = text.upper().strip().replace(" ", "")
            if conf >= self._min_confidence and text:
                results.append((text, float(conf)))
                logger.debug("EasyOCR read: '%s'  conf=%.2f", text, conf)

        results.sort(key=lambda x: x[1], reverse=True)
        return results


 
# PaddleOCR backend
 

class PaddleOCREngine(OCREngine):
    """
    PaddleOCR-backed OCR engine.

    Parameters
    ----------
    lang : str
        Language code ('en' for English).
    use_gpu : bool
        Whether to use GPU inference.
    min_confidence : float
        Drop reads below this threshold.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        use_angle_cls: bool = False,
        min_confidence: float = 0.45,
    ) -> None:
        self._min_confidence = min_confidence

        try:
            import os
            os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
            os.environ["FLAGS_use_mkldnn"] = "0"
            os.environ["FLAGS_use_onednn"] = "0"
            os.environ["PADDLE_PDX_CPU_NUM_THREADS"] = "4"
            from paddleocr import PaddleOCR  # type: ignore

            logger.info(
                "Initialising PaddleOCR — lang=%s  gpu=%s", lang, use_gpu
            )
            # PaddleOCR v3+ removed show_log / use_gpu / use_angle_cls args.
            # GPU is controlled by PaddlePaddle internally; pass only supported params.
            import inspect
            supported = inspect.signature(PaddleOCR.__init__).parameters
            kwargs: dict = {"lang": lang}
            if "use_angle_cls" in supported:        # v2 compat
                kwargs["use_angle_cls"] = use_angle_cls
            if "use_gpu" in supported:              # v2 compat
                kwargs["use_gpu"] = use_gpu
            if "show_log" in supported:             # v2 compat
                kwargs["show_log"] = False
            
            # CPU/Inference Speed Optimizations (Skipping redundant preprocessing models)
            if "use_textline_orientation" in supported:
                kwargs["use_textline_orientation"] = False
            if "use_doc_orientation_classify" in supported:
                kwargs["use_doc_orientation_classify"] = False
            if "use_doc_unwarping" in supported:
                kwargs["use_doc_unwarping"] = False
            if "text_det_limit_side_len" in supported:
                kwargs["text_det_limit_side_len"] = 320 # Resize 960 -> 320 for rapid detection

            self._reader = PaddleOCR(**kwargs)
            logger.info("PaddleOCR ready (Optimized for speed).")
        except ImportError as exc:
            raise ImportError(
                "paddleocr is not installed. Run: pip install paddleocr"
            ) from exc

    # ------------------------------------------------------------------

    def read(self, image: np.ndarray) -> Tuple[str, float]:
        candidates = self.read_all(image)
        if not candidates:
            return "", 0.0
        return candidates[0]

    def read_all(self, image: np.ndarray) -> List[Tuple[str, float]]:
        if image is None or image.size == 0:
            return []

        raw = self._reader.ocr(image)

        results: List[Tuple[str, float]] = []
        if not raw or not raw[0]:
            return results

        ocr_res = raw[0]
        if hasattr(ocr_res, "get"):
            # Modern PaddleOCR / PaddleX format (OCRResult dict)
            rec_texts = ocr_res.get("rec_texts", [])
            rec_scores = ocr_res.get("rec_scores", [])
            for text, conf in zip(rec_texts, rec_scores):
                clean_text = str(text).upper().strip().replace(" ", "")
                conf_val = float(conf)
                if conf_val >= self._min_confidence and clean_text:
                    results.append((clean_text, conf_val))
                    logger.debug("PaddleOCR read: '%s'  conf=%.2f", clean_text, conf_val)
        else:
            # Legacy PaddleOCR format (list-of-lists)
            for line in ocr_res:
                if line is None:
                    continue
                text_conf = line[1]
                clean_text = str(text_conf[0]).upper().strip().replace(" ", "")
                conf_val = float(text_conf[1])
                if conf_val >= self._min_confidence and clean_text:
                    results.append((clean_text, conf_val))
                    logger.debug("PaddleOCR read: '%s'  conf=%.2f", clean_text, conf_val)

        results.sort(key=lambda x: x[1], reverse=True)
        return results


 
# Factory
 

def build_ocr_engine(cfg: dict) -> OCREngine:
    """
    Construct the OCR engine from a config dict (the 'ocr' section of config.yaml).

    Example cfg::

        {
            "backend": "easyocr",
            "languages": ["en"],
            "gpu": True,
            "min_confidence": 0.45,
            "easyocr": {"allowlist": "ABC...", ...},
        }
    """
    backend = cfg.get("backend", "easyocr").lower()
    gpu = cfg.get("gpu", True)
    min_conf = cfg.get("min_confidence", 0.45)

    if backend == "easyocr":
        eocr_cfg = cfg.get("easyocr", {})
        return EasyOCREngine(
            languages=cfg.get("languages", ["en"]),
            gpu=gpu,
            allowlist=eocr_cfg.get(
                "allowlist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ),
            min_confidence=min_conf,
        )

    if backend == "paddleocr":
        paddle_cfg = cfg.get("paddleocr", {})
        langs = cfg.get("languages", ["en"])
        lang = langs[0] if langs else "en"
        return PaddleOCREngine(
            lang=lang,
            use_gpu=gpu,
            use_angle_cls=paddle_cfg.get("use_angle_cls", False),
            min_confidence=min_conf,
        )

    raise ValueError(
        f"Unknown OCR backend '{backend}'. Choose 'easyocr' or 'paddleocr'."
    )
