"""Логи этапов для скриптов в run/: время начала и длительность каждого шага."""

from __future__ import annotations

import time
from contextlib import contextmanager


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} c"
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes} мин {seconds} c"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@contextmanager
def step(message: str):
    """Логирует начало и конец шага:

        with step("строим индекс BM25"):
            ...
    """
    log(f"▶ {message}")
    t0 = time.time()
    yield
    log(f"✓ {message} — {_duration(time.time() - t0)}")
