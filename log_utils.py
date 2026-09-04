import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_dir_name: str = "logs") -> logging.Logger:
    """Create a logger that appends to a single file for that script.

    The log filename stays fixed (for example, train.log), while each log line
    includes an automatic timestamp from the logging formatter. This means every
    new training or testing run adds to the same file instead of replacing it.
    """
    base_dir = Path(__file__).resolve().parent
    logs_dir = base_dir / log_dir_name
    logs_dir.mkdir(exist_ok=True)

    log_path = logs_dir / f"{name}.log"

    logger = logging.getLogger(f"transformer_logger_{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Log file opened: %s", log_path)
    return logger
