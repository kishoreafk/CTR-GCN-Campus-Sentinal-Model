"""Unified logger: console (coloured) + rotating file + JSON lines."""
import logging, json, sys
from datetime import datetime
from pathlib import Path

COLOURS = {"DEBUG": "\033[36m", "INFO": "\033[32m",
           "WARNING": "\033[33m", "ERROR": "\033[31m", "RESET": "\033[0m"}

class ColourFormatter(logging.Formatter):
    def format(self, record):
        c = COLOURS.get(record.levelname, "")
        r = COLOURS["RESET"]
        record.msg = f"{c}{record.msg}{r}"
        return super().format(record)

def setup_logger(name: str, log_dir: str = "logs",
                 level: str = "INFO") -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(ColourFormatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))

    fh = logging.FileHandler(f"{log_dir}/{name}_{ts}.log")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
