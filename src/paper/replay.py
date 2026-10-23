"""Replays recorded sessions through the paper broker.

Rows come from the recorder's Parquet files, which hold both the features and
the raw 20-level book, so fills use the same depth the live engine saw.
Probabilities are computed for all rows in one batch (same model and same
signal rule as live), then each row is fed to the broker in time order.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Final

import polars as pl

from ..feed.binary_parser import ExchangeSegment
from ..features.calculator import MODEL_FEATURES
from ..inference.predictor import SignalPredictor, signal_for
from .broker import PaperBroker

PROGRESS_EVERY: Final = 2_000


def _session(col: str = "recv_ns") -> pl.Expr:
    return (
        pl.from_epoch(col, time_unit="ns")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Asia/Kolkata")
        .dt.date()
    )


def _parquet_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.parquet"))


def recorded_sessions(data_dir: Path) -> list[dict]:
    """[{date, exchange_segment, security_id, rows}] for everything recorded under data_dir."""
    files = _parquet_files(data_dir)
    if not files:
        return []
    df = (
        pl.scan_parquet(files)
        .select("recv_ns", "exchange_segment", "security_id")
        .with_columns(session=_session())
        .group_by("session", "exchange_segment", "security_id")
        .len()
        .sort("session", "security_id")
        .collect()
    )
    return [
        {"date": s.isoformat(), "exchange_segment": ExchangeSegment(seg).name, "security_id": sid, "rows": n}
        for s, seg, sid, n in df.iter_rows()
    ]


def load_replay_frame(data_dir: Path, security_id: int, days: list[date]) -> pl.DataFrame:
    files = _parquet_files(data_dir)
    if not files:
        raise ValueError(f"No recordings under {data_dir}")
    lf = (
        pl.scan_parquet(files)
        .filter(pl.col("security_id") == security_id)
        .with_columns(session=_session())
    )
    if days:
        lf = lf.filter(pl.col("session").is_in(days))
    df = lf.select("recv_ns", *MODEL_FEATURES, "bid_px", "bid_qty", "ask_px", "ask_qty").sort("recv_ns").collect()
    if df.is_empty():
        wanted = ", ".join(d.isoformat() for d in days) or "any day"
        raise ValueError(f"No recorded rows for security {security_id} on {wanted}")
    return df


def run_replay(
    broker: PaperBroker,
    predictor: SignalPredictor,
    frame: pl.DataFrame,
    cancel: threading.Event,
    progress: Callable[[float], None],
) -> bool:
    """Feed every row to the broker. Returns False if cancelled part-way."""
    proba = predictor.predict_batch(frame.select(MODEL_FEATURES).to_numpy())
    threshold = predictor.threshold
    signals = [signal_for(p_down, p_up, threshold) for p_down, p_up in zip(proba[:, 0].tolist(), proba[:, 2].tolist())]

    ts = frame["recv_ns"].to_list()
    bid_px = frame["bid_px"].to_numpy()
    bid_qty = frame["bid_qty"].to_numpy()
    ask_px = frame["ask_px"].to_numpy()
    ask_qty = frame["ask_qty"].to_numpy()
    n = len(ts)

    for i in range(n):
        broker.on_tick(ts[i], bid_px[i], bid_qty[i], ask_px[i], ask_qty[i], signals[i])
        if i % PROGRESS_EVERY == 0:
            progress(i / n)
            if cancel.is_set():
                broker.finish("cancelled")
                return False
    broker.finish()
    progress(1.0)
    return True
