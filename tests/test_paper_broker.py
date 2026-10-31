"""PaperBroker on hand-built books: fills, charges, exits and risk limits with exact expected numbers."""

from datetime import date, datetime, time

import numpy as np
import pytest

from src.paper.broker import PaperBroker, order_charges, walk_book
from src.paper.config import Charges, ExperimentConfig
from src.pipeline.recorder import IST

DAY = date(2026, 10, 20)


def at(hh: int, mm: int, ss: float = 0.0, day: date = DAY) -> int:
    base = datetime.combine(day, time(hh, mm), tzinfo=IST).timestamp()
    return int((base + ss) * 1e9)


def book(best_bid: float, qty: int = 10_000):
    """20 levels, 0.05 apart, ``qty`` shares each; best ask one tick above best bid."""
    steps = np.arange(20) * 0.05
    bid_px, ask_px = np.round(best_bid - steps, 2), np.round(best_bid + 0.05 + steps, 2)
    q = np.full(20, qty, dtype=np.int64)
    return bid_px, q, ask_px, q.copy()


def cfg(**overrides) -> ExperimentConfig:
    base = dict(
        mode="replay", security_id=1333, model="m.txt", starting_capital=100_000,
        size_mode="value", trade_size=50_000, charges=Charges(enabled=False),
        daily_loss_limit=0, hold_seconds=3.0,
    )
    return ExperimentConfig(**(base | overrides))


def tick(b: PaperBroker, ts: int, best_bid: float, signal: str, qty: int = 10_000) -> None:
    b.on_tick(ts, *book(best_bid, qty), signal)


# ---- pure helpers -----------------------------------------------------------------

def test_walk_book_consumes_levels_in_order():
    px, q = np.array([100.0, 100.05, 100.10, 0.0]), np.array([100, 100, 100, 0])
    assert walk_book(px, q, 250) == (250, pytest.approx(100 * 100.0 + 100 * 100.05 + 50 * 100.10))
    assert walk_book(px, q, 500) == (300, pytest.approx(30_015.0))  # partial: book exhausted
    filled, turnover = walk_book(px, q, 500, force=True)  # exits assume the rest fills at the worst level
    assert filled == 500 and turnover == pytest.approx(30_015.0 + 200 * 100.10)


def test_order_charges_match_hand_calculation():
    c = Charges()
    # ₹1,00,000: brokerage min(20, 30) = 20; exchange 2.97; SEBI 0.10; GST 18% of 23.07 = 4.1526.
    assert order_charges(c, "buy", 100_000) == pytest.approx(20 + 2.97 + 0.10 + 4.1526 + 3.0)   # + stamp 0.003%
    assert order_charges(c, "sell", 100_000) == pytest.approx(20 + 2.97 + 0.10 + 4.1526 + 25.0)  # + STT 0.025%
    assert order_charges(Charges(enabled=False), "sell", 100_000) == 0.0
    # Small orders pay 0.03% brokerage, below the ₹20 cap.
    assert order_charges(c, "buy", 10_000) == pytest.approx(3.0 + 0.297 + 0.01 + 0.18 * 3.307 + 0.3)


# ---- trading behaviour -------------------------------------------------------------

def test_long_round_trip_with_time_exit():
    b = PaperBroker(cfg())
    tick(b, at(10, 0), 100.00, "LONG")                    # buy at ask 100.05
    assert b.position is not None and b.position.qty == 499  # 50,000 // 100.05
    tick(b, at(10, 0, 2.9), 100.20, "FLAT")
    assert b.position is not None                          # hold not yet elapsed
    tick(b, at(10, 0, 3.0), 100.20, "FLAT")                # time exit at bid 100.20
    (t,) = b.trades
    assert (t.side, t.exit_reason) == ("LONG", "time")
    assert t.entry_price == pytest.approx(100.05) and t.exit_price == pytest.approx(100.20)
    assert t.gross_pnl == pytest.approx(0.15 * 499)
    assert b.cash == pytest.approx(100_000 + 0.15 * 499)


def test_short_profits_when_price_falls():
    b = PaperBroker(cfg())
    tick(b, at(10, 0), 100.00, "SHORT")                   # sell at bid 100.00
    tick(b, at(10, 0, 3), 99.80, "FLAT")                  # cover at ask 99.85
    (t,) = b.trades
    assert t.side == "SHORT" and t.qty == 500
    assert t.gross_pnl == pytest.approx((100.00 - 99.85) * 500)


def test_opposite_signal_exits_and_reverses_on_the_same_tick():
    b = PaperBroker(cfg())
    tick(b, at(10, 0), 100.00, "LONG")
    tick(b, at(10, 0, 1), 100.00, "SHORT")
    assert b.trades[0].exit_reason == "opposite signal"
    assert b.position is not None and b.position.side == -1


def test_shorts_can_be_disabled():
    b = PaperBroker(cfg(allow_short=False))
    tick(b, at(10, 0), 100.00, "SHORT")
    assert b.position is None and not b.trades


def test_entries_wait_for_start_time():
    b = PaperBroker(cfg(start_time=time(9, 30)))
    tick(b, at(9, 20), 100.00, "LONG")
    assert b.position is None
    tick(b, at(9, 30), 100.00, "LONG")
    assert b.position is not None


def test_square_off_closes_and_blocks_new_entries():
    b = PaperBroker(cfg(hold_seconds=0))  # only the opposite signal or square-off can close
    tick(b, at(15, 0), 100.00, "LONG")
    tick(b, at(15, 15), 100.10, "LONG")
    assert b.trades[0].exit_reason == "square-off" and b.position is None
    tick(b, at(15, 20), 100.10, "LONG")
    assert b.position is None and b.halted


def test_daily_loss_limit_halts_until_next_day():
    b = PaperBroker(cfg(daily_loss_limit=300))
    tick(b, at(10, 0), 100.00, "LONG")                    # 499 shares
    tick(b, at(10, 0, 1), 99.00, "LONG")                  # mark at bid 99.00: about -524
    assert b.trades[0].exit_reason == "loss limit" and b.halted
    tick(b, at(11, 0), 99.00, "LONG")
    assert b.position is None                              # still halted today
    tick(b, at(10, 0, day=date(2026, 10, 21)), 99.00, "LONG")
    assert b.position is not None and not b.halted         # fresh day


def test_latency_delays_fill_to_a_later_book():
    b = PaperBroker(cfg(latency_ms=500))
    tick(b, at(10, 0), 100.00, "LONG")
    tick(b, at(10, 0, 0.2), 100.10, "LONG")
    assert b.position is None                              # order still in flight
    tick(b, at(10, 0, 0.6), 100.30, "LONG")
    assert b.position is not None and b.position.entry_price == pytest.approx(100.35)


def test_size_is_capped_by_equity_and_leverage():
    b = PaperBroker(cfg(size_mode="shares", trade_size=5_000))
    tick(b, at(10, 0), 100.00, "LONG")
    assert b.position.qty == 999                           # 100,000 // 100.05 at 1x
    b2 = PaperBroker(cfg(size_mode="shares", trade_size=5_000, leverage=5))
    tick(b2, at(10, 0), 100.00, "LONG")
    assert b2.position.qty == 4_997                        # 500,000 // 100.05


def test_large_orders_walk_the_book():
    b = PaperBroker(cfg(size_mode="shares", trade_size=250))
    tick(b, at(10, 0), 100.00, "LONG", qty=100)            # 100 @ 100.05, 100 @ 100.10, 50 @ 100.15
    assert b.position.entry_price == pytest.approx((100 * 100.05 + 100 * 100.10 + 50 * 100.15) / 250)


def test_cooldown_spaces_out_entries():
    b = PaperBroker(cfg(cooldown_seconds=10))
    tick(b, at(10, 0), 100.00, "LONG")
    tick(b, at(10, 0, 3), 100.00, "LONG")                 # time exit; cooldown starts
    assert b.position is None
    tick(b, at(10, 0, 8), 100.00, "LONG")
    assert b.position is None
    tick(b, at(10, 0, 13), 100.00, "LONG")
    assert b.position is not None


def test_session_end_flattens_and_finish_flattens():
    b = PaperBroker(cfg(hold_seconds=0))
    tick(b, at(14, 0), 100.00, "LONG")
    tick(b, at(9, 30, day=date(2026, 10, 21)), 101.00, "FLAT")  # data ended before square-off
    assert b.trades[0].exit_reason == "session end"
    tick(b, at(9, 31, day=date(2026, 10, 21)), 101.00, "LONG")
    b.finish()
    assert b.trades[-1].exit_reason == "experiment end" and b.position is None


def test_metrics_reconcile_with_charges():
    b = PaperBroker(cfg(charges=Charges()))
    for i, (bid, sig) in enumerate([(100.0, "LONG"), (100.3, "FLAT"), (100.3, "SHORT"), (100.1, "FLAT")]):
        tick(b, at(10, 0, i * 3), bid, sig)
    b.finish()
    m = b.metrics()
    assert m["trades"] == 2 and m["charges"] > 0
    assert m["net_pnl"] == pytest.approx(m["gross_pnl"] - m["charges"])
    assert m["final_equity"] == pytest.approx(100_000 + sum(t.net_pnl for t in b.trades))
    assert [d["trades"] for d in b.days()] == [2]


def test_crossed_book_is_ignored():
    b = PaperBroker(cfg())
    bid_px, bq, ask_px, aq = book(100.0)
    b.on_tick(at(10, 0), ask_px, bq, bid_px, aq, "LONG")  # bids above asks
    assert b.position is None and b.ticks == 0


def test_config_validation():
    with pytest.raises(ValueError):
        cfg(start_time=time(15, 20))                        # after square-off
    with pytest.raises(ValueError):
        cfg(model="../secret.txt")
    with pytest.raises(ValueError):
        cfg(size_mode="shares", trade_size=10.5)
    with pytest.raises(ValueError):
        cfg(signal_threshold=0.4)
    with pytest.raises(ValueError, match="cannot exceed"):
        cfg(size_mode="value", trade_size=150_000, starting_capital=100_000, leverage=1.0)
