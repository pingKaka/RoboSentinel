from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar


T = TypeVar("T")


class TimeSeriesBuffer(Generic[T]):
    """A bounded, monotonic timestamp buffer for online observations."""

    def __init__(self, duration_sec: float, timestamp: Callable[[T], float]) -> None:
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        self.duration_sec = duration_sec
        self.timestamp = timestamp
        self._items: deque[T] = deque()

    def append(self, item: T) -> None:
        value = self.timestamp(item)
        if self._items and value < self.timestamp(self._items[-1]):
            raise ValueError("TimeSeriesBuffer requires monotonic timestamps")
        self._items.append(item)
        cutoff = value - self.duration_sec
        while self._items and self.timestamp(self._items[0]) < cutoff:
            self._items.popleft()

    def nearest(self, timestamp_sec: float) -> tuple[T | None, float]:
        if not self._items:
            return None, float("inf")
        item = min(self._items, key=lambda candidate: abs(self.timestamp(candidate) - timestamp_sec))
        return item, abs(self.timestamp(item) - timestamp_sec)

    def items(self) -> tuple[T, ...]:
        return tuple(self._items)

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)

    def __len__(self) -> int:
        return len(self._items)
