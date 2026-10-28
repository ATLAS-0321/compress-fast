"""Diagnostics: one logger, counters and timers for every public step."""
import logging
import threading
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_PATH = Path.home() / ".hermes" / "logs" / "compress-fast.log"

log = logging.getLogger("compress_fast")

_lock = threading.Lock()
_counters: dict[str, int] = {}
_timers: dict[str, list[float]] = {}  # name -> [count, total_ms, max_ms]


def setup(path: Path = LOG_PATH, level: int = logging.INFO) -> None:
    if log.handlers:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(level)


def count(name: str, n: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + n


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000
        with _lock:
            entry = _timers.setdefault(name, [0, 0.0, 0.0])
            entry[0] += 1
            entry[1] += ms
            entry[2] = max(entry[2], ms)


def stats() -> dict:
    with _lock:
        return {
            "counters": dict(_counters),
            "timers_ms": {k: {"count": v[0], "total": round(v[1], 2), "max": round(v[2], 2),
                              "mean": round(v[1] / v[0], 2) if v[0] else 0.0}
                          for k, v in _timers.items()},
        }
