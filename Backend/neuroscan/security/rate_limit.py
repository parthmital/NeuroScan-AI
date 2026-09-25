"""Sliding-window, per-client request limiter for authentication endpoints.

State is per process. Account lockout (stored in the database) is the cross-process control;
this limiter only slows credential stuffing against a single instance.
"""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            if len(self._hits) > 10_000:
                self._evict(now)
            return True

    def _evict(self, now: float) -> None:
        stale = [
            key for key, hits in self._hits.items() if not hits or now - hits[-1] >= self.window
        ]
        for key in stale:
            del self._hits[key]
