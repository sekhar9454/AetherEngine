"""Simulated intraday broker for paper trading with fake money.

Driven purely by book updates and their timestamps, never the wall clock, so a
replayed day and a live day go through exactly the same logic.

Fills walk the real 20-level book: a buy consumes ask levels from the best
price outward, a sell consumes bids. One position at a time, closed by a time
exit, an opposite signal, the daily loss limit or the square-off time.
Positions never carry overnight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Final, Literal

import numpy as np

from .config import Charges, ExperimentConfig

IST_OFFSET_NS: Final = 19_800 * 10**9  # +05:30; India has no daylight saving
DAY_NS: Final = 86_400 * 10**9
EQUITY_SAMPLE_NS: Final = 5 * 10**9


def ist_day(ns: int) -> int:
    return (ns + IST_OFFSET_NS) // DAY_NS


def ist_seconds(ns: int) -> float:
    return ((ns + IST_OFFSET_NS) % DAY_NS) / 1e9


def day_to_date(day: int) -> date:
    return date(1970, 1, 1) + timedelta(days=day)


def order_charges(c: Charges, side: Literal["buy", "sell"], turnover: float) -> float:
    if not c.enabled:
        return 0.0
    brokerage = min(c.brokerage_cap, turnover * c.brokerage_pct / 100)
    exchange = turnover * c.exchange_pct / 100
    sebi = turnover * c.sebi_per_crore / 1e7
    gst = (brokerage + exchange + sebi) * c.gst_pct / 100
    tax = turnover * (c.stt_sell_pct if side == "sell" else c.stamp_buy_pct) / 100
    return brokerage + exchange + sebi + gst + tax


def walk_book(prices: np.ndarray, quantities: np.ndarray, want: int, force: bool = False) -> tuple[int, float]:
    """Fill up to ``want`` shares from best level outward; returns (filled qty, turnover).

    With ``force``, any quantity beyond the visible 20 levels is assumed to fill
    at the worst visible price, so exits always flatten the position.
    """
    remaining, filled, turnover, worst = want, 0, 0.0, 0.0
    for p, q in zip(prices.tolist(), quantities.tolist()):
        if p <= 0 or q <= 0:
            continue
        take = min(int(q), remaining)
        turnover += take * p
        filled += take
        remaining -= take
        worst = p
        if remaining == 0:
            break
    if remaining and force and worst > 0:
        turnover += remaining * worst
        filled += remaining
    return filled, turnover


@dataclass(slots=True)
class Position:
    side: int  # +1 long, -1 short
    qty: int
    entry_price: float
    entry_ns: int
    entry_charges: float


@dataclass(slots=True)
class Trade:
    side: str
    qty: int
    entry_ns: int
    entry_price: float
    exit_ns: int
    exit_price: float
    gross_pnl: float
    charges: float
    net_pnl: float
    exit_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class _Order:
    kind: Literal["entry", "exit"]
    side: int  # entry: direction of the new position
    reason: str
    due_ns: int


class PaperBroker:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.cash = cfg.starting_capital
        self.position: Position | None = None
        self.pending: _Order | None = None
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.day_stats: dict[int, dict[str, float]] = {}
        self.ticks = 0
        self.halted = False  # no new entries for the rest of the day
        self.halt_reason = ""
        self.peak_equity = cfg.starting_capital
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

        self._day: int | None = None
        self._day_start_equity = cfg.starting_capital
        self._last_exit_ns: int | None = None
        self._last_sample_ns = 0
        self._last_ns = 0
        self._book: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

        self._hold_ns = int(cfg.hold_seconds * 1e9)
        self._cooldown_ns = int(cfg.cooldown_seconds * 1e9)
        self._latency_ns = int(cfg.latency_ms * 1e6)
        self._start_s = cfg.start_time.hour * 3600 + cfg.start_time.minute * 60 + cfg.start_time.second
        sq = cfg.square_off_time
        self._square_off_s = sq.hour * 3600 + sq.minute * 60 + sq.second

    # ---- state ---------------------------------------------------------------

    def equity(self) -> float:
        pos = self.position
        if pos is None or self._book is None:
            return self.cash
        bid_px, _, ask_px, _ = self._book
        # Marked at the price it could be closed at right now, not the mid.
        mark = bid_px[0] if pos.side > 0 else ask_px[0]
        return self.cash + pos.side * pos.qty * float(mark)

    # ---- event loop ------------------------------------------------------------

    def on_tick(
        self,
        ts: int,
        bid_px: np.ndarray,
        bid_qty: np.ndarray,
        ask_px: np.ndarray,
        ask_qty: np.ndarray,
        signal: str,
    ) -> None:
        if bid_px[0] <= 0 or ask_px[0] <= 0 or bid_px[0] >= ask_px[0]:
            return
        day = ist_day(ts)
        if day != self._day:
            self._new_day(day)
        self._book = (bid_px, bid_qty, ask_px, ask_qty)
        self._last_ns = ts
        self.ticks += 1

        if self.pending is not None and ts >= self.pending.due_ns:
            self._execute(self.pending, ts)

        equity = self.equity()
        self._track_drawdown(equity)
        limit = self.cfg.daily_loss_limit
        if limit > 0 and not self.halted and equity - self._day_start_equity <= -limit:
            self._halt("daily loss limit")
            if self.position is not None:
                self._close(ts, "loss limit")
            self.pending = None

        seconds = ist_seconds(ts)
        if seconds >= self._square_off_s:
            if not self.halted:
                self._halt("square-off time")
            if self.position is not None and (self.pending is None or self.pending.kind != "exit"):
                self._order_exit(ts, "square-off")
        elif self.position is not None and self.pending is None:
            pos = self.position
            if self._hold_ns and ts - pos.entry_ns >= self._hold_ns:
                self._order_exit(ts, "time")
            elif self.cfg.exit_on_opposite and signal == ("SHORT" if pos.side > 0 else "LONG"):
                self._order_exit(ts, "opposite signal")

        if (
            self.position is None
            and self.pending is None
            and not self.halted
            and seconds >= self._start_s
            and (self._last_exit_ns is None or ts - self._last_exit_ns >= self._cooldown_ns)
        ):
            if signal == "LONG":
                self._order_entry(ts, +1)
            elif signal == "SHORT" and self.cfg.allow_short:
                self._order_entry(ts, -1)

        if ts - self._last_sample_ns >= EQUITY_SAMPLE_NS:
            self._sample(ts)

    def finish(self, reason: str = "experiment end") -> None:
        """Flatten any open position at the last seen book and record final equity."""
        self.pending = None
        if self.position is not None and self._book is not None:
            self._close(self._last_ns, reason)
        if self._last_ns:
            self._sample(self._last_ns)

    # ---- internals -------------------------------------------------------------

    def _new_day(self, day: int) -> None:
        if self.position is not None and self._book is not None:
            self._close(self._last_ns, "session end")  # data ended before square-off
        self.pending = None
        self._day = day
        self._day_start_equity = self.cash
        self.halted = False
        self.halt_reason = ""
        self._last_exit_ns = None

    def _halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def _order_entry(self, ts: int, side: int) -> None:
        order = _Order("entry", side, "signal", ts + self._latency_ns)
        if self._latency_ns:
            self.pending = order
        else:
            self._execute(order, ts)

    def _order_exit(self, ts: int, reason: str) -> None:
        order = _Order("exit", 0, reason, ts + self._latency_ns)
        if self._latency_ns:
            self.pending = order
        else:
            self._execute(order, ts)

    def _execute(self, order: _Order, ts: int) -> None:
        self.pending = None
        if order.kind == "exit":
            if self.position is not None:
                self._close(ts, order.reason)
        elif self.position is None and not self.halted:
            self._open(ts, order.side)

    def _open(self, ts: int, side: int) -> None:
        bid_px, bid_qty, ask_px, ask_qty = self._book
        price = float(ask_px[0] if side > 0 else bid_px[0])
        if self.cfg.size_mode == "shares":
            qty = int(self.cfg.trade_size)
        else:
            qty = int(self.cfg.trade_size // price)
        qty = min(qty, int(max(self.equity(), 0.0) * self.cfg.leverage // price))
        if qty <= 0:
            return
        if side > 0:
            filled, turnover = walk_book(ask_px, ask_qty, qty)
            if not filled:
                return
            charges = order_charges(self.cfg.charges, "buy", turnover)
            self.cash -= turnover + charges
        else:
            filled, turnover = walk_book(bid_px, bid_qty, qty)
            if not filled:
                return
            charges = order_charges(self.cfg.charges, "sell", turnover)
            self.cash += turnover - charges
        self.position = Position(side, filled, turnover / filled, ts, charges)

    def _close(self, ts: int, reason: str) -> None:
        pos = self.position
        bid_px, bid_qty, ask_px, ask_qty = self._book
        if pos.side > 0:
            _, turnover = walk_book(bid_px, bid_qty, pos.qty, force=True)
            charges = order_charges(self.cfg.charges, "sell", turnover)
            self.cash += turnover - charges
        else:
            _, turnover = walk_book(ask_px, ask_qty, pos.qty, force=True)
            charges = order_charges(self.cfg.charges, "buy", turnover)
            self.cash -= turnover + charges
        exit_price = turnover / pos.qty
        gross = pos.side * (exit_price - pos.entry_price) * pos.qty
        total_charges = pos.entry_charges + charges
        trade = Trade(
            side="LONG" if pos.side > 0 else "SHORT",
            qty=pos.qty,
            entry_ns=pos.entry_ns,
            entry_price=pos.entry_price,
            exit_ns=ts,
            exit_price=exit_price,
            gross_pnl=gross,
            charges=total_charges,
            net_pnl=gross - total_charges,
            exit_reason=reason,
        )
        self.trades.append(trade)
        self.position = None
        self._last_exit_ns = ts
        stats = self.day_stats.setdefault(ist_day(ts), {"trades": 0, "net_pnl": 0.0, "charges": 0.0})
        stats["trades"] += 1
        stats["net_pnl"] += trade.net_pnl
        stats["charges"] += total_charges
        self._sample(ts)

    def _track_drawdown(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = self.peak_equity - equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_drawdown_pct = dd / self.peak_equity * 100

    def _sample(self, ts: int) -> None:
        equity = self.equity()
        self._track_drawdown(equity)
        self.equity_curve.append((ts, equity))
        self._last_sample_ns = ts

    # ---- reporting -------------------------------------------------------------

    def metrics(self) -> dict:
        # list() snapshots are atomic under the GIL, so this is safe to call while
        # a replay thread is still appending.
        trades = list(self.trades)
        nets = [t.net_pnl for t in trades]
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n <= 0]
        equity = self.equity()
        capital = self.cfg.starting_capital
        hold = [(t.exit_ns - t.entry_ns) / 1e9 for t in trades]
        return {
            "final_equity": equity,
            "net_pnl": equity - capital,
            "return_pct": (equity - capital) / capital * 100,
            "gross_pnl": sum(t.gross_pnl for t in trades),
            "charges": sum(t.charges for t in trades),
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else None,
            "avg_net_per_trade": sum(nets) / len(nets) if nets else None,
            "profit_factor": sum(wins) / -sum(losses) if losses and sum(losses) < 0 else None,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_hold_s": sum(hold) / len(hold) if hold else None,
            "long_trades": sum(1 for t in trades if t.side == "LONG"),
            "short_trades": sum(1 for t in trades if t.side == "SHORT"),
            "ticks": self.ticks,
        }

    def days(self) -> list[dict]:
        return [
            {"date": day_to_date(day).isoformat(), **stats}
            for day, stats in sorted(list(self.day_stats.items()))
        ]

    def open_position(self) -> dict | None:
        pos = self.position
        if pos is None or self._book is None:
            return None
        bid_px, _, ask_px, _ = self._book
        mark = float(bid_px[0] if pos.side > 0 else ask_px[0])
        return {
            "side": "LONG" if pos.side > 0 else "SHORT",
            "qty": pos.qty,
            "entry_price": pos.entry_price,
            "entry_ns": pos.entry_ns,
            "mark_price": mark,
            "unrealized_pnl": pos.side * (mark - pos.entry_price) * pos.qty - pos.entry_charges,
        }
