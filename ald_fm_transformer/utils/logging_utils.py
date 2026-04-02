from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterable

from .io import ensure_dir


def make_logger(log_path: str | Path) -> logging.Logger:
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.fieldnames = list(fieldnames)
        if not self.path.exists():
            with self.path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict):
        with self.path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
