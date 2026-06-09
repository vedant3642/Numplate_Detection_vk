"""
utils/logger.py
Structured, rotating file + console logger for the pipeline.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


_initialized: bool = False
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "numplate") -> logging.Logger:
    """Return (and lazily initialise) the root pipeline logger."""
    global _logger, _initialized
    if _initialized and _logger is not None:
        return logging.getLogger(name)
    return logging.getLogger(name)


def setup_logger(
    level: str = "INFO",
    log_file: str = "logs/pipeline.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 3,
) -> logging.Logger:
    """
    Configure the root 'numplate' logger with a console handler and a
    rotating file handler.  Call this once from run.py before anything else.
    """
    global _initialized, _logger

    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("numplate")
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(numeric_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _initialized = True
    _logger = logger
    return logger
