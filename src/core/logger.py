import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

# 1. Get the environment (Defaults to 'development' if not set)
# In production, you would set an environment variable: export APP_ENV=production
APP_ENV = os.getenv("APP_ENV", "development").lower()

# 2. Directory for logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 3. Log Formats
# Development: Human-readable, detailed
DEV_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
# Production: Strict, easy to parse, includes process ID
PROD_FORMAT = "%(asctime)s | %(levelname)-8s | %(process)d | %(name)s | %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class ColoredFormatter(logging.Formatter):
    """Adds colors to log levels for console output (Development only)."""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)
        self.fmt = fmt
        self.formats = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.fmt, # Default color for info
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.
    Behavior changes based on APP_ENV environment variable.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # ==========================================
    # DEVELOPMENT CONFIGURATION
    # ==========================================
    if APP_ENV == "development":
        logger.setLevel(logging.DEBUG) # Catch everything

        # Console Handler (Colored)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(DEV_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler (Rotating)
        # 'midnight' creates a new file every day at 00:00
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(LOG_DIR, "app.log"),
            when="midnight",
            interval=1,
            backupCount=7, # Keep logs for 7 days
            encoding="utf-8"
        )
        file_handler.suffix = "%Y-%m-%d.log" # Naming pattern: rag_app.log.2023-10-27.log
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(DEV_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)

    # ==========================================
    # PRODUCTION CONFIGURATION
    # ==========================================
    else:
        logger.setLevel(logging.INFO) # Only important stuff

        # No Console Output in Production (usually cluttering for servers)
        # Or you can keep it simple:
        # console_handler = logging.StreamHandler(sys.stdout)
        # console_handler.setLevel(logging.INFO)
        # logger.addHandler(console_handler)

        # File Handler (Rotating)
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(LOG_DIR, "production.log"),
            when="midnight",
            interval=1,
            backupCount=90, # Keep logs for 30 days in prod
            encoding="utf-8"
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.setLevel(logging.INFO)
        # Using the stricter PROD_FORMAT
        file_handler.setFormatter(logging.Formatter(PROD_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger