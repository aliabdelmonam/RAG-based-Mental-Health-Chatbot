"""
Logging setup.

Development  → DEBUG to stdout (colored) + rotating file under logs/
Production   → INFO to stdout only (let the infra capture it)

Set APP_ENV=production to activate production mode.
"""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from core.Config import get_settings
# ── Constants ─────────────────────────────────────────────────────────────────

_DEV_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_PROD_FMT = "%(asctime)s | %(levelname)-8s | %(process)d | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_COLORS = {
    logging.DEBUG:    "\x1b[38;20m",   # grey
    logging.WARNING:  "\x1b[33;20m",   # yellow
    logging.ERROR:    "\x1b[31;20m",   # red
    logging.CRITICAL: "\x1b[31;1m",    # bold red
}
_RESET = "\x1b[0m"

settings = get_settings()

# ── Colored formatter ─────────────────────────────────────────────────────────

class _ColoredFormatter(logging.Formatter):
    """Colorizes log-level prefix for console output in development."""

    def format(self, record: logging.LogRecord) -> str:
        color  = _COLORS.get(record.levelno, "")
        record = logging.makeLogRecord(record.__dict__)   # shallow copy — don't mutate original
        record.levelname = f"{color}{record.levelname}{_RESET}" if color else record.levelname
        return super().format(record)


# ── Handler factories ─────────────────────────────────────────────────────────

def _console_handler(level: int, colored: bool) -> logging.Handler:
    handler   = logging.StreamHandler(sys.stdout)
    formatter = _ColoredFormatter(_DEV_FMT, datefmt=_DATE_FMT) if colored \
                else logging.Formatter(_PROD_FMT, datefmt=_DATE_FMT)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _file_handler(path: str, level: int, fmt: str, backup_days: int) -> logging.Handler:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=path,
        when="midnight",
        interval=1,
        backupCount=backup_days,
        encoding="utf-8",
        delay=True,      # don't open the file until the first log record
    )
    handler.suffix = "%Y-%m-%d"
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FMT))
    return handler


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, configured once.
    Subsequent calls with the same name return the cached instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    env = settings.APP_ENV.lower()
    is_prod = env == "production"

    logger.setLevel(logging.INFO if is_prod else logging.DEBUG)
    logger.propagate = False     # don't double-log if a root handler exists

    if is_prod:
        # Stdout only — Docker/systemd/k8s capture and rotate for you.
        logger.addHandler(_console_handler(logging.INFO, colored=False))
    else:
        logger.addHandler(_console_handler(logging.DEBUG, colored=True))
        logger.addHandler(_file_handler(
            path="logs/app.log",
            level=logging.DEBUG,
            fmt=_DEV_FMT,
            backup_days=7,
        ))

    return logger