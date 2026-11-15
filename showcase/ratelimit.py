"""In-memory token-bucket rate limiting keyed by client IP.

Serverless note: each Vercel instance has its own memory, so this is a best-effort brake against a
runaway loop or a casual abuser, not a hard guarantee. Pair it with provider-side spend limits.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from fastapi import Request

OVERALL = (30, 60.0)   # 30 requests per 60 s across the API
HEAVY = (8, 60.0)      # 8 per 60 s on routes that call a metered model
POLL = (90, 60.0)      # managed-agent event polling has its own generous bucket
MAX_KEYS = 5000


class RateLimited(Exception):
    def __init__(self, retry_after: int) -> None:
        super().__init__("rate limited")
        self.retry_after = max(1, retry_after)


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: OrderedDict[tuple[str, str], _Bucket] = OrderedDict()

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def _refill(self, ip: str, name: str, limit: tuple[int, float], now: float) -> _Bucket:
        cap, window = limit
        b = self._buckets.get((ip, name))
        if b is None:
            b = self._buckets[(ip, name)] = _Bucket(float(cap), now)
        self._buckets.move_to_end((ip, name))
        b.tokens = min(float(cap), b.tokens + (now - b.updated) * cap / window)
        b.updated = now
        return b

    def check(self, ip: str, kinds: tuple[str, ...] = ("overall",), now: float | None = None) -> None:
        """Take one token from every named bucket, or raise RateLimited without taking any."""
        limits = {"overall": OVERALL, "heavy": HEAVY, "poll": POLL}
        now = time.monotonic() if now is None else now
        with self._lock:
            buckets = [(self._refill(ip, k, limits[k], now), limits[k]) for k in kinds]
            wait = 0.0
            for b, (cap, window) in buckets:
                if b.tokens < 1:
                    wait = max(wait, (1 - b.tokens) * window / cap)
            if wait:
                raise RateLimited(math.ceil(wait))
            for b, _ in buckets:
                b.tokens -= 1
            while len(self._buckets) > MAX_KEYS:  # hard bound: evict the least recently used entries
                self._buckets.popitem(last=False)


def client_ip(request: Request) -> str:
    """The rate-limit key. On Vercel the platform sets x-vercel-forwarded-for / x-real-ip (a client cannot spoof them
    there); anywhere else only the socket peer is trusted. A client-supplied X-Forwarded-For is never used."""
    if os.getenv("VERCEL") == "1":
        for name in ("x-vercel-forwarded-for", "x-real-ip"):
            first = request.headers.get(name, "").split(",")[0].strip()
            if first and len(first) <= 45:
                return first
    return request.client.host if request.client else "unknown"
