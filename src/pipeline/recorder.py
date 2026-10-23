"""Persists every computed feature vector X_t to Parquet via polars.

Parquet files cannot be appended to, so rows are buffered in memory and flushed
as part files under ``<data_dir>/<IST date>/part-<first recv_ns>.parquet``. A
crash loses at most one buffer. Alongside the features (which include
mid_price), the raw 20-level book is stored so features can be re-derived later.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from ..features.calculator import FEATURE_NAMES
from ..features.orderbook import OrderBook

logger = logging.getLogger(__name__)

IST: Final = ZoneInfo("Asia/Kolkata")
BOOK_COLUMNS: Final = ("bid_px", "bid_qty", "ask_px", "ask_qty")


class _Buffer:
    __slots__ = ("recv_ns", "segment", "security_id", "features", "books")

    def __init__(self) -> None:
        self.recv_ns: list[int] = []
        self.segment: list[int] = []
        self.security_id: list[int] = []
        self.features: list[np.ndarray] = []
        self.books: dict[str, list[np.ndarray]] = {c: [] for c in BOOK_COLUMNS}

    def __len__(self) -> int:
        return len(self.recv_ns)


class ParquetRecorder:
    def __init__(self, data_dir: Path, flush_rows: int = 50_000) -> None:
        self.data_dir = data_dir
        self.flush_rows = flush_rows
        self.rows_written = 0
        self.files_written = 0
        self._buffer = _Buffer()
        self._write_lock = asyncio.Lock()

    @property
    def pending(self) -> int:
        return len(self._buffer)

    def append(self, recv_ns: int, book: OrderBook, features: np.ndarray) -> None:
        buf = self._buffer
        buf.recv_ns.append(recv_ns)
        buf.segment.append(book.exchange_segment)
        buf.security_id.append(book.security_id)
        # Copies: the book's arrays are overwritten in place on the next update.
        buf.features.append(features.copy())
        buf.books["bid_px"].append(book.bid_px.copy())
        buf.books["bid_qty"].append(book.bid_qty.copy())
        buf.books["ask_px"].append(book.ask_px.copy())
        buf.books["ask_qty"].append(book.ask_qty.copy())

    @property
    def should_flush(self) -> bool:
        return len(self._buffer) >= self.flush_rows

    async def flush(self) -> None:
        """Write the current buffer in a worker thread without blocking the event loop."""
        buf = self._swap()
        if buf is None:
            return
        async with self._write_lock:
            await asyncio.to_thread(self._write, buf)

    def close(self) -> None:
        """Synchronously write whatever is left; call on shutdown."""
        buf = self._swap()
        if buf is not None:
            self._write(buf)

    def _swap(self) -> _Buffer | None:
        if not self._buffer:
            return None
        buf, self._buffer = self._buffer, _Buffer()
        return buf

    def _write(self, buf: _Buffer) -> None:
        features = np.vstack(buf.features)
        frame = pl.DataFrame(
            {
                "recv_ns": pl.Series(buf.recv_ns, dtype=pl.Int64),
                "exchange_segment": pl.Series(buf.segment, dtype=pl.UInt8),
                "security_id": pl.Series(buf.security_id, dtype=pl.UInt32),
                **{name: features[:, i] for i, name in enumerate(FEATURE_NAMES)},
                **{c: pl.Series(c, np.vstack(buf.books[c])) for c in BOOK_COLUMNS},
            }
        )
        # Partition by IST date so folders line up with trading sessions, even when
        # a buffer spans midnight.
        first_day = datetime.fromtimestamp(buf.recv_ns[0] / 1e9, IST).date()
        last_day = datetime.fromtimestamp(buf.recv_ns[-1] / 1e9, IST).date()
        if first_day == last_day:
            self._write_part(frame, first_day.isoformat())
            return
        day = (
            pl.from_epoch("recv_ns", time_unit="ns")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Asia/Kolkata")
            .dt.date()
            .alias("_day")
        )
        for (part_day,), part in frame.with_columns(day).partition_by("_day", as_dict=True).items():
            self._write_part(part.drop("_day"), part_day.isoformat())

    def _write_part(self, frame: pl.DataFrame, day: str) -> None:
        out_dir = self.data_dir / day
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"part-{frame['recv_ns'][0]}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        try:
            frame.write_parquet(tmp, compression="zstd")
            tmp.replace(path)  # atomic: readers never see a half-written file
        except Exception:
            logger.exception("Failed to write %d rows to %s", len(frame), path)
            tmp.unlink(missing_ok=True)
            return
        self.rows_written += len(frame)
        self.files_written += 1
        logger.info("Recorded %d rows -> %s", len(frame), path)
