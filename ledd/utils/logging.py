"""Console + CSV logging. Kaggle/Colab sessions die often, so everything is appended
to disk immediately rather than buffered."""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from typing import Any, Dict


def get_logger(name: str = "ledd", logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class CSVLogger:
    """Append-only metric log. Survives session kills; safe to resume into."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fields: list[str] | None = None
        if os.path.exists(path):
            with open(path) as f:
                r = csv.reader(f)
                header = next(r, None)
                if header:
                    self._fields = header

    def log(self, row: Dict[str, Any]) -> None:
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        if self._fields is None:
            self._fields = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self._fields)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self._fields])
