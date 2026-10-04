"""Logging to stdout and a file, plus an append-only CSV for per-epoch metrics."""
import csv
import logging
import sys
from pathlib import Path


def setup_logging(log_file: Path | None = None, enabled: bool = True) -> None:
    """Send log records to stdout and, optionally, a file.

    Args:
        log_file: File to also log to; parent folders are created.
        enabled: False raises the level to WARNING, used to silence non-main DDP ranks.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO if enabled else logging.WARNING, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


class CSVLogger:
    """One row per write(); header written on first write to a new file, missing fields left empty."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames

    def write(self, row: dict) -> None:
        new_file = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})
