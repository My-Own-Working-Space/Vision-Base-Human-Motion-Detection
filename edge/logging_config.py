"""
Structured logging configuration for the Edge Device.

Replaces all print() statements with proper Python logging.
Uses structured format with timestamps, levels, and module context.
"""

from __future__ import annotations

import logging
import sys


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure structured logging for the entire application.

    Args:
        level: Root logging level.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates on reload
    root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(console)

    # Mute noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a specific module.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
