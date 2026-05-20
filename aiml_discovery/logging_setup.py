"""Logging configuration for aiml_discovery.

Call configure_logging() once at application startup (e.g. in app.py).
All sub-modules use logging.getLogger(__name__), which automatically
inherits from the 'aiml_discovery' root logger configured here.

Log levels:
  Console  — INFO and above
  Log file — DEBUG and above (full detail for troubleshooting)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "aiml_gui.log"
_LOGGER_NAME = "aiml_discovery"

_FMT = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def configure_logging(console_level: int = logging.INFO) -> None:
    """Set up console + rotating-file handlers on the aiml_discovery logger.

    Safe to call multiple times — skips setup if handlers are already attached.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(_FMT)
    logger.addHandler(ch)

    # Rotating file handler — 5 MB per file, keep 3 backups
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_FMT)
        logger.addHandler(fh)
        logger.debug("File logging active → %s", _LOG_FILE)
    except OSError as exc:
        logger.warning("Could not open log file %s: %s — file logging disabled.", _LOG_FILE, exc)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the aiml_discovery namespace."""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if not name.startswith(_LOGGER_NAME) else name)
