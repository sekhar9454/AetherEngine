"""Microstructure feature vector X_t computed from an L2 book snapshot.

Features (in FEATURE_NAMES order):
    spread, spread_bps          best ask - best bid, absolute and relative to mid
    mid_price, micro_price      (ask0 + bid0) / 2 and the queue-weighted micro-price
    micro_offset_bps            micro-price minus mid, in bps of mid
    obi_l1, obi_5, obi_20       (bid - ask) / (bid + ask) quantity over top 1 / 5 / 20 levels
    {bid,ask}_band_{1,2,3}      resting volume within 0-0.05%, 0.05-0.15%, 0.15-0.50% of mid
    band_{1,2,3}_imbalance      (bid_band - ask_band) / (bid_band + ask_band)

Absolute price levels (mid_price, micro_price) are recorded and displayed but
excluded from MODEL_FEATURES: they drift from day to day, so a tree model
would learn price regimes rather than book shape.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from .orderbook import OrderBook

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "spread", "spread_bps", "mid_price", "micro_price", "micro_offset_bps",
    "obi_l1", "obi_5", "obi_20",
    "bid_band_1", "bid_band_2", "bid_band_3",
    "ask_band_1", "ask_band_2", "ask_band_3",
    "band_1_imbalance", "band_2_imbalance", "band_3_imbalance",
)
FEATURE_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(FEATURE_NAMES)}

MODEL_FEATURES: Final[tuple[str, ...]] = tuple(
    n for n in FEATURE_NAMES if n not in ("mid_price", "micro_price")
)
MODEL_FEATURE_INDEX: Final = np.array([FEATURE_INDEX[n] for n in MODEL_FEATURES], dtype=np.intp)

# Upper edges of the spectrum bands as a fraction of mid; anything further out is ignored.
BAND_EDGES: Final = np.array([0.0005, 0.0015, 0.0050])
OBI_LEVELS: Final = 5


def _imbalance(bid: float, ask: float) -> float:
    total = bid + ask
    return (bid - ask) / total if total > 0 else 0.0


def _spectrum(distance: np.ndarray, qty: np.ndarray) -> np.ndarray:
    # searchsorted(side="left") maps d to band i where EDGES[i-1] < d <= EDGES[i];
    # index len(EDGES) collects everything beyond the last band (incl. empty levels).
    band = np.searchsorted(BAND_EDGES, distance, side="left")
    return np.bincount(band, weights=qty, minlength=len(BAND_EDGES) + 1)[: len(BAND_EDGES)]


def compute_features(book: OrderBook) -> np.ndarray | None:
    """Return X_t aligned with FEATURE_NAMES, or None if the book is not tradeable
    (a side missing, or a locked/crossed top of book as seen in auctions)."""
    best_bid = float(book.bid_px[0])
    best_ask = float(book.ask_px[0])
    if not book.ready or best_bid <= 0.0 or best_ask <= 0.0 or best_bid >= best_ask:
        return None

    bid_qty = book.bid_qty.astype(np.float64)
    ask_qty = book.ask_qty.astype(np.float64)
    bid_q0, ask_q0 = bid_qty[0], ask_qty[0]

    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    top_qty = bid_q0 + ask_q0
    micro = (bid_q0 * best_ask + ask_q0 * best_bid) / top_qty if top_qty > 0 else mid

    bid_bands = _spectrum((mid - book.bid_px) / mid, bid_qty)
    ask_bands = _spectrum((book.ask_px - mid) / mid, ask_qty)

    x = np.empty(len(FEATURE_NAMES), dtype=np.float64)
    x[0] = spread
    x[1] = spread / mid * 1e4
    x[2] = mid
    x[3] = micro
    x[4] = (micro - mid) / mid * 1e4
    x[5] = _imbalance(bid_q0, ask_q0)
    x[6] = _imbalance(bid_qty[:OBI_LEVELS].sum(), ask_qty[:OBI_LEVELS].sum())
    x[7] = _imbalance(bid_qty.sum(), ask_qty.sum())
    x[8:11] = bid_bands
    x[11:14] = ask_bands
    for i in range(len(BAND_EDGES)):
        x[14 + i] = _imbalance(bid_bands[i], ask_bands[i])
    return x
