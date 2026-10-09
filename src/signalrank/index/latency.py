"""Measuring what a query costs under load.

Open loop rather than closed loop. A closed loop issues the next query when the
last one returns, so a slow index simply receives fewer queries and its
percentiles flatter it. Here the arrival rate is fixed and queries are issued on
schedule whether or not the previous ones have finished, which is how a real
system meets traffic and the only way a latency figure at a stated rate means
anything.

Queueing delay counts. If the index cannot keep up, the wait shows up in the
numbers rather than being absorbed by a slower arrival rate.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyReport:
    queries: int
    target_qps: float
    achieved_qps: float
    seconds: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "queries": self.queries,
            "target_qps": self.target_qps,
            "achieved_qps": round(self.achieved_qps, 1),
            "seconds": round(self.seconds, 2),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def measure(
    search: Callable[[np.ndarray], object],
    queries: np.ndarray,
    *,
    target_qps: float,
    seconds: float,
    workers: int = 16,
) -> LatencyReport:
    """Issue queries on a fixed schedule and record how long each took.

    Latency is measured from when a query was *due* rather than from when a
    worker picked it up, so time spent waiting for a free worker is counted
    against the index. Measuring from pickup would hide exactly the behaviour
    that matters when the arrival rate exceeds what the index can serve.
    """
    interval = 1.0 / target_qps
    total = int(target_qps * seconds)
    latencies: list[float] = []
    lock = threading.Lock()
    schedule = iter(range(total))
    started = time.perf_counter()

    def worker() -> None:
        while True:
            with lock:
                index = next(schedule, None)
            if index is None:
                return
            due = started + index * interval
            now = time.perf_counter()
            if now < due:
                time.sleep(due - now)
            search(queries[index % len(queries)][None, :])
            with lock:
                latencies.append((time.perf_counter() - due) * 1000.0)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started

    samples = np.asarray(latencies, dtype=np.float64)
    return LatencyReport(
        queries=len(latencies),
        target_qps=target_qps,
        achieved_qps=len(latencies) / elapsed,
        seconds=elapsed,
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        p99_ms=float(np.percentile(samples, 99)),
        max_ms=float(samples.max()),
    )
