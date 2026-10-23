"""Starts, tracks, persists and stops paper-trading experiments.

Replay experiments run in a worker thread over recorded Parquet data. Live
experiments attach to the running engine as a sink and trade on every feature
update for their instrument. Both use PaperBroker, so results are comparable.

Each experiment is saved under ``<experiments dir>/<id>/`` as experiment.json
(settings, metrics, equity curve, recent trades) plus trades.csv (every trade).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import secrets
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import polars as pl

from ..feed.binary_parser import ExchangeSegment
from ..features.orderbook import BookKey, OrderBook
from ..inference.predictor import Prediction, SignalPredictor
from ..pipeline.recorder import IST
from .broker import PaperBroker, Trade
from .config import ExperimentConfig
from .replay import load_replay_frame, recorded_sessions, run_replay

if TYPE_CHECKING:
    from ..feed.websocket_client import FeedStats, Instrument

logger = logging.getLogger(__name__)

MAX_EQUITY_POINTS: Final = 1_500
SUMMARY_KEYS: Final = (
    "id", "name", "mode", "status", "progress", "error", "created", "finished",
    "exchange_segment", "security_id", "model",
    "net_pnl", "return_pct", "trades", "win_rate", "charges", "max_drawdown_pct",
)
RECENT_TRADES: Final = 200
AUTOSAVE_S: Final = 30.0


def _now() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def _ist(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, IST).isoformat(timespec="milliseconds")


def _trade_json(t: Trade) -> dict:
    return {
        "side": t.side,
        "qty": t.qty,
        "entry_ms": t.entry_ns // 1_000_000,
        "entry_price": t.entry_price,
        "exit_ms": t.exit_ns // 1_000_000,
        "exit_price": t.exit_price,
        "gross_pnl": t.gross_pnl,
        "charges": t.charges,
        "net_pnl": t.net_pnl,
        "exit_reason": t.exit_reason,
    }


def trades_csv(trades: list[Trade]) -> bytes:
    frame = pl.DataFrame(
        {
            "side": [t.side for t in trades],
            "qty": [t.qty for t in trades],
            "entry_time": [_ist(t.entry_ns) for t in trades],
            "entry_price": [round(t.entry_price, 4) for t in trades],
            "exit_time": [_ist(t.exit_ns) for t in trades],
            "exit_price": [round(t.exit_price, 4) for t in trades],
            "hold_s": [round((t.exit_ns - t.entry_ns) / 1e9, 3) for t in trades],
            "gross_pnl": [round(t.gross_pnl, 2) for t in trades],
            "charges": [round(t.charges, 2) for t in trades],
            "net_pnl": [round(t.net_pnl, 2) for t in trades],
            "exit_reason": [t.exit_reason for t in trades],
        },
        schema_overrides={"qty": pl.Int64},
    )
    buf = io.BytesIO()
    frame.write_csv(buf)
    return buf.getvalue()


def _downsample(points: list[tuple[int, float]]) -> list[list[float]]:
    if len(points) > MAX_EQUITY_POINTS:
        step = math.ceil(len(points) / MAX_EQUITY_POINTS)
        points = points[::step] + [points[-1]]
    return [[ts // 1_000_000, round(eq, 2)] for ts, eq in points]


class Experiment:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.id = f"{datetime.now(IST):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        self.config = cfg
        self.name = cfg.name or (
            f"{cfg.mode} {cfg.security_id} @ {cfg.signal_threshold:.2f}, "
            f"hold {cfg.hold_seconds:g}s"
        )
        self.created = _now()
        self.finished: str | None = None
        self.status = "running"
        self.progress = 0.0
        self.error: str | None = None
        self.broker = PaperBroker(cfg)
        self.last_prediction: Prediction | None = None

    def summary(self) -> dict[str, Any]:
        m = self.broker.metrics()
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.config.mode,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "created": self.created,
            "finished": self.finished,
            "exchange_segment": self.config.exchange_segment,
            "security_id": self.config.security_id,
            "model": self.config.model,
            "net_pnl": m["net_pnl"],
            "return_pct": m["return_pct"],
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "charges": m["charges"],
            "max_drawdown_pct": m["max_drawdown_pct"],
        }

    def detail(self) -> dict[str, Any]:
        broker = self.broker
        pred = self.last_prediction
        return {
            **self.summary(),
            "config": self.config.model_dump(mode="json"),
            "metrics": broker.metrics(),
            "days": broker.days(),
            "equity": _downsample(list(broker.equity_curve)),
            "recent_trades": [_trade_json(t) for t in reversed(list(broker.trades)[-RECENT_TRADES:])],
            "position": broker.open_position(),
            "halted": broker.halt_reason if broker.halted else None,
            "prediction": None if pred is None else {
                "p_down": pred.p_down, "p_flat": pred.p_flat, "p_up": pred.p_up, "signal": pred.signal,
            },
        }


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def save(self, detail: dict, trades: list[Trade]) -> None:
        folder = self.root / detail["id"]
        folder.mkdir(exist_ok=True)
        for name, payload in (
            ("experiment.json", json.dumps(detail, indent=1).encode()),
            ("trades.csv", trades_csv(trades)),
        ):
            tmp = folder / f"{name}.tmp"
            tmp.write_bytes(payload)
            tmp.replace(folder / name)

    def load_all(self) -> dict[str, dict]:
        saved: dict[str, dict] = {}
        for path in sorted(self.root.glob("*/experiment.json")):
            try:
                detail = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable experiment %s", path)
                continue
            if detail.get("status") == "running":
                # Process exited without finalizing it (crash or kill).
                detail["status"] = "interrupted"
            saved[detail["id"]] = detail
        return saved

    def trades_csv_path(self, exp_id: str) -> Path:
        return self.root / exp_id / "trades.csv"

    def delete(self, exp_id: str) -> None:
        shutil.rmtree(self.root / exp_id, ignore_errors=True)


class LiveSink:
    """Engine sink: runs one live experiment on its instrument's feature updates."""

    def __init__(self, exp: Experiment, predictor: SignalPredictor, key: BookKey, manager: ExperimentManager) -> None:
        self.exp = exp
        self.predictor = predictor
        self.key = key
        self.manager = manager

    def on_features(self, book: OrderBook, x: np.ndarray) -> None:
        if book.key != self.key:
            return
        try:
            pred = self.predictor.predict_proba(x)
            self.exp.last_prediction = pred
            self.exp.broker.on_tick(
                book.last_update_ns, book.bid_px, book.bid_qty, book.ask_px, book.ask_qty, pred.signal
            )
        except Exception as exc:  # one broken experiment must not take down the feed
            logger.exception("Live experiment %s failed", self.exp.id)
            self.manager.fail_live(self.exp.id, exc)


class ExperimentManager:
    def __init__(
        self,
        store: ExperimentStore,
        data_dir: Path,
        models_dir: Path,
        engine: Any | None = None,
        instruments: tuple[Instrument, ...] = (),
        feed_stats: FeedStats | None = None,
    ) -> None:
        self.store = store
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.engine = engine
        self.instruments = instruments
        self.feed_stats = feed_stats
        self._active: dict[str, Experiment] = {}
        self._saved = store.load_all()
        self._replays: dict[str, tuple[asyncio.Task, threading.Event]] = {}
        self._sinks: dict[str, LiveSink] = {}

    # ---- discovery -------------------------------------------------------------

    def models(self) -> list[str]:
        """LightGBM model files (they start with a 'tree' line) in the models directory."""
        found = []
        for path in sorted(self.models_dir.glob("*.txt")):
            try:
                with path.open() as fh:
                    if fh.readline().strip() == "tree":
                        found.append(path.name)
            except OSError:
                continue
        return found

    def meta(self) -> dict[str, Any]:
        stats = self.feed_stats
        return {
            "models": self.models(),
            "sessions": recorded_sessions(self.data_dir),
            "live": {
                "available": self.engine is not None,
                "connected": bool(stats and stats.connected),
                "instruments": [
                    {"exchange_segment": i.exchange_segment, "security_id": int(i.security_id)}
                    for i in self.instruments
                ],
            },
            "defaults": ExperimentConfig.model_construct(mode="replay", security_id=0, model="")
            .model_dump(mode="json") | {"security_id": None, "model": None},
        }

    # ---- lifecycle -------------------------------------------------------------

    async def start(self, cfg: ExperimentConfig) -> str:
        if cfg.model not in self.models():
            raise ValueError(f"Model {cfg.model!r} not found in {self.models_dir}")
        predictor = SignalPredictor(self.models_dir / cfg.model, cfg.signal_threshold)

        if cfg.mode == "replay":
            frame = await asyncio.to_thread(load_replay_frame, self.data_dir, cfg.security_id, cfg.days)
            exp = Experiment(cfg)
            cancel = threading.Event()
            task = asyncio.create_task(self._run_replay(exp, predictor, frame, cancel))
            self._active[exp.id] = exp
            self._replays[exp.id] = (task, cancel)
        else:
            if self.engine is None:
                raise ValueError("Live mode needs the Dhan feed; start the engine without --no-feed")
            key = (ExchangeSegment[cfg.exchange_segment].value, cfg.security_id)
            subscribed = {(ExchangeSegment[i.exchange_segment].value, int(i.security_id)) for i in self.instruments}
            if key not in subscribed:
                raise ValueError(
                    f"{cfg.exchange_segment}:{cfg.security_id} is not subscribed; add it to DHAN_INSTRUMENTS"
                )
            exp = Experiment(cfg)
            sink = LiveSink(exp, predictor, key, self)
            self._active[exp.id] = exp
            self._sinks[exp.id] = sink
            self.engine.sinks.append(sink)
        logger.info("Started %s experiment %s (%s)", cfg.mode, exp.id, exp.name)
        return exp.id

    async def _run_replay(
        self, exp: Experiment, predictor: SignalPredictor, frame: pl.DataFrame, cancel: threading.Event
    ) -> None:
        def progress(p: float) -> None:
            exp.progress = p

        try:
            done = await asyncio.to_thread(run_replay, exp.broker, predictor, frame, cancel, progress)
            exp.status = "completed" if done else "stopped"
        except Exception as exc:
            logger.exception("Replay %s failed", exp.id)
            exp.status, exp.error = "failed", str(exc)
        finally:
            self._replays.pop(exp.id, None)
            self._finalize(exp)

    async def stop(self, exp_id: str) -> bool:
        if exp_id in self._replays:
            task, cancel = self._replays[exp_id]
            cancel.set()
            await asyncio.wait_for(asyncio.shield(task), timeout=30)
            return True
        if exp_id in self._sinks:
            exp = self._detach(exp_id)
            exp.broker.finish("stopped")
            exp.status = "stopped"
            self._finalize(exp)
            return True
        return False

    def fail_live(self, exp_id: str, exc: Exception) -> None:
        exp = self._detach(exp_id)
        exp.status, exp.error = "failed", str(exc)
        self._finalize(exp)

    def _detach(self, exp_id: str) -> Experiment:
        sink = self._sinks.pop(exp_id)
        if sink in self.engine.sinks:
            self.engine.sinks.remove(sink)
        return sink.exp

    def _finalize(self, exp: Experiment) -> None:
        exp.finished = _now()
        detail = exp.detail()
        try:
            self.store.save(detail, list(exp.broker.trades))
        except OSError:
            logger.exception("Could not save experiment %s", exp.id)
        self._saved[exp.id] = detail
        self._active.pop(exp.id, None)
        logger.info("Experiment %s %s: net P&L %.2f over %d trades",
                    exp.id, exp.status, detail["net_pnl"], detail["trades"])

    async def autosave(self) -> None:
        """Periodically persist running live experiments so a crash loses little."""
        while True:
            await asyncio.sleep(AUTOSAVE_S)
            for exp_id in list(self._sinks):
                exp = self._active.get(exp_id)
                if exp is not None:
                    await asyncio.to_thread(self.store.save, exp.detail(), list(exp.broker.trades))

    async def shutdown(self) -> None:
        for exp_id in list(self._active):
            try:
                await self.stop(exp_id)
            except Exception:
                logger.exception("Failed to stop experiment %s", exp_id)

    # ---- queries ---------------------------------------------------------------

    def list(self) -> list[dict]:
        summaries = [e.summary() for e in self._active.values()]
        summaries += [{k: d.get(k) for k in SUMMARY_KEYS} for d in self._saved.values()]
        return sorted(summaries, key=lambda s: s["created"], reverse=True)

    def get(self, exp_id: str) -> dict | None:
        if exp_id in self._active:
            return self._active[exp_id].detail()
        return self._saved.get(exp_id)

    def trades_csv(self, exp_id: str) -> bytes | None:
        if exp_id in self._active:
            return trades_csv(list(self._active[exp_id].broker.trades))
        path = self.store.trades_csv_path(exp_id)
        return path.read_bytes() if exp_id in self._saved and path.exists() else None

    def delete(self, exp_id: str) -> bool:
        if exp_id in self._active:
            raise RuntimeError("Stop the experiment before deleting it")
        if self._saved.pop(exp_id, None) is None:
            return False
        self.store.delete(exp_id)
        return True
