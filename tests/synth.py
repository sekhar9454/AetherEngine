"""Test helpers: Dhan packet builders and a synthetic market with a planted signal.

In SynthMarket, a hidden AR(1) state drives both the book imbalance and the
drift of the mid-price, so book shape genuinely predicts the next few moves.
"""

from __future__ import annotations

import math
import struct

import numpy as np

TICK = 0.05


def depth_packet(code: int, security_id: int, seq: int, base: float, segment: int = 1) -> bytes:
    levels = b"".join(struct.pack("<dII", base + i * TICK, 100 + i, 1 + i) for i in range(20))
    return struct.pack("<hBBII", 332, code, segment, security_id, seq) + levels


def disconnect_packet(reason: int) -> bytes:
    return struct.pack("<hBBII", 14, 50, 1, 0, 0) + struct.pack("<h", reason)


class SynthMarket:
    def __init__(self, seed: int = 0, security_id: int = 1333, mid: float = 1500.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.security_id = security_id
        self.mid = mid
        self.s = 0.0
        self.seq = 0

    def _step(self):
        r = self.rng
        self.s = 0.97 * self.s + 0.25 * r.standard_normal()
        self.mid *= 1 + (0.25 * self.s + 0.6 * r.standard_normal()) / 1e4
        best_bid = math.floor(self.mid / TICK) * TICK
        best_ask = best_bid + TICK * (1 if r.random() < 0.8 else 2)
        skew = math.exp(0.6 * self.s)
        bid_px = np.round(np.sort(best_bid - TICK * np.arange(20) * (1 + r.integers(0, 3, 20)))[::-1], 2)
        ask_px = np.round(np.sort(best_ask + TICK * np.arange(20) * (1 + r.integers(0, 3, 20))), 2)
        bid_q = (r.integers(50, 500, 20) * skew).astype(np.int64)
        ask_q = (r.integers(50, 500, 20) / skew).astype(np.int64)
        return bid_px, bid_q, r.integers(1, 20, 20), ask_px, ask_q, r.integers(1, 20, 20)

    def updates(self, recv_ns: int) -> list[dict]:
        """The next bid and ask as parsed DepthUpdate dicts."""
        bp, bq, bo, ap, aq, ao = self._step()
        self.seq += 2
        return [
            {"kind": "depth", "side": side, "exchange_segment": 1, "security_id": self.security_id,
             "sequence": seq, "recv_ns": recv_ns,
             "prices": p.tolist(), "quantities": q.tolist(), "orders": o.tolist()}
            for side, p, q, o, seq in (("bid", bp, bq, bo, self.seq - 1), ("ask", ap, aq, ao, self.seq))
        ]

    def frame(self) -> bytes:
        """The next bid and ask as one binary WebSocket frame, as Dhan stacks them."""
        bp, bq, bo, ap, aq, ao = self._step()
        self.seq += 2
        out = b""
        for code, p, q, o in ((41, bp, bq, bo), (51, ap, aq, ao)):
            out += struct.pack("<hBBII", 332, code, 1, self.security_id, self.seq)
            out += b"".join(struct.pack("<dII", float(p[i]), int(q[i]), int(o[i])) for i in range(20))
        return out
