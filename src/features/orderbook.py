"""In-memory state of the 20-level L2 book.

Dhan sends each side as a full 20-level snapshot in its own packet, so applying
an update is a straight overwrite of that side; there are no incremental deltas.
"""

from __future__ import annotations

import numpy as np

from ..feed.binary_parser import DEPTH_LEVELS, DepthUpdate

BookKey = tuple[int, int]  # (exchange_segment, security_id)


class OrderBook:
    __slots__ = (
        "exchange_segment", "security_id",
        "bid_px", "bid_qty", "bid_orders", "ask_px", "ask_qty", "ask_orders",
        "bid_ns", "ask_ns", "updates",
    )

    def __init__(self, exchange_segment: int, security_id: int) -> None:
        self.exchange_segment = exchange_segment
        self.security_id = security_id
        # Level 0 is the best price. Empty levels hold price 0 and quantity 0.
        self.bid_px = np.zeros(DEPTH_LEVELS, dtype=np.float64)
        self.bid_qty = np.zeros(DEPTH_LEVELS, dtype=np.int64)
        self.bid_orders = np.zeros(DEPTH_LEVELS, dtype=np.int64)
        self.ask_px = np.zeros(DEPTH_LEVELS, dtype=np.float64)
        self.ask_qty = np.zeros(DEPTH_LEVELS, dtype=np.int64)
        self.ask_orders = np.zeros(DEPTH_LEVELS, dtype=np.int64)
        self.bid_ns = 0
        self.ask_ns = 0
        self.updates = 0

    @property
    def key(self) -> BookKey:
        return (self.exchange_segment, self.security_id)

    @property
    def ready(self) -> bool:
        """Both sides have been received at least once."""
        return self.bid_ns > 0 and self.ask_ns > 0

    @property
    def last_update_ns(self) -> int:
        return max(self.bid_ns, self.ask_ns)

    def apply(self, update: DepthUpdate) -> None:
        if update["side"] == "bid":
            self.bid_px[:] = update["prices"]
            self.bid_qty[:] = update["quantities"]
            self.bid_orders[:] = update["orders"]
            self.bid_ns = update["recv_ns"]
        else:
            self.ask_px[:] = update["prices"]
            self.ask_qty[:] = update["quantities"]
            self.ask_orders[:] = update["orders"]
            self.ask_ns = update["recv_ns"]
        self.updates += 1


class BookRegistry:
    """Books for every subscribed instrument, created on first update."""

    def __init__(self) -> None:
        self._books: dict[BookKey, OrderBook] = {}

    def apply(self, update: DepthUpdate) -> OrderBook:
        key = (update["exchange_segment"], update["security_id"])
        book = self._books.get(key)
        if book is None:
            book = self._books[key] = OrderBook(*key)
        book.apply(update)
        return book

    def get(self, key: BookKey) -> OrderBook | None:
        return self._books.get(key)

    def __iter__(self):
        return iter(self._books.values())

    def __len__(self) -> int:
        return len(self._books)
