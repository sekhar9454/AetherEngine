"""Asyncio orchestrator and Rich terminal dashboard.

    python -m src.main --record                 # collect training data
    python -m src.main --model model.txt        # live signals
    python -m src.main --model model.txt --record --headless
    python -m src.main --portal                 # paper-trading lab at http://127.0.0.1:8000
    python -m src.main --portal --no-feed       # replay-only lab, no Dhan credentials needed

Credentials and instruments come from .env (see FeedConfig.from_env).
Pipeline per frame: WebSocket -> binary parser -> queue -> order book ->
features -> (recorder, predictor) -> dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .feed.binary_parser import DisconnectNotice, ExchangeSegment, FeedMessage
from .feed.websocket_client import DepthFeedClient, FeedConfig, FeedFatalError, FeedStats
from .features.calculator import FEATURE_INDEX as F
from .features.calculator import compute_features
from .features.orderbook import BookKey, BookRegistry, OrderBook
from .inference.predictor import Prediction, Signal, SignalPredictor
from .paper.manager import ExperimentManager, ExperimentStore
from .pipeline.recorder import IST, ParquetRecorder
from .portal.server import bind, create_app, make_server

logger = logging.getLogger("engine")

LADDER_LEVELS: Final = 5
LATENCY_WINDOW: Final = 2_000


class FeatureSink(Protocol):
    """Receives every tradeable book update with its feature vector (e.g. a live paper experiment)."""

    def on_features(self, book: OrderBook, x: np.ndarray) -> None: ...


@dataclass(slots=True)
class InstrumentState:
    book: OrderBook
    features: np.ndarray | None = None
    prediction: Prediction | None = None
    signal: Signal = "FLAT"


@dataclass(frozen=True, slots=True)
class SignalEvent:
    recv_ns: int
    security_id: int
    signal: Signal
    probability: float


class Engine:
    """Consumes feed messages, maintains books, computes features, records and predicts."""

    def __init__(
        self,
        queue: asyncio.Queue[FeedMessage],
        predictor: SignalPredictor | None,
        recorder: ParquetRecorder | None,
    ) -> None:
        self.queue = queue
        self.predictor = predictor
        self.recorder = recorder
        self.books = BookRegistry()
        self.states: dict[BookKey, InstrumentState] = {}
        self.signals: deque[SignalEvent] = deque(maxlen=8)
        self.last_disconnect: DisconnectNotice | None = None
        self.sinks: list[FeatureSink] = []
        self.feature_updates = 0
        self.untradeable = 0  # crossed/locked/one-sided books skipped
        self.calc_us: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self.infer_us: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self.e2e_us: deque[float] = deque(maxlen=LATENCY_WINDOW)

    async def run(self) -> None:
        while True:
            batch = [await self.queue.get()]
            self._drain_into(batch)
            self._process_batch(batch)

    def drain(self) -> None:
        """Process whatever is still queued; used on shutdown before the final flush."""
        batch: list[FeedMessage] = []
        self._drain_into(batch)
        self._process_batch(batch)

    def _drain_into(self, batch: list[FeedMessage]) -> None:
        while True:
            try:
                batch.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return

    def _process_batch(self, batch: list[FeedMessage]) -> None:
        # Apply every queued packet first, then compute once per changed book, so a
        # frame's bid and ask packets are never evaluated half-applied and a backlog
        # is conflated to the latest state instead of replayed.
        dirty: dict[BookKey, OrderBook] = {}
        for msg in batch:
            if msg["kind"] == "depth":
                book = self.books.apply(msg)
                dirty[book.key] = book
            else:
                self.last_disconnect = msg
        for book in dirty.values():
            self._on_book(book)

    def _on_book(self, book: OrderBook) -> None:
        start = time.perf_counter_ns()
        x = compute_features(book)
        self.calc_us.append((time.perf_counter_ns() - start) / 1e3)
        if x is None:
            self.untradeable += 1
            return

        state = self.states.get(book.key)
        if state is None:
            state = self.states[book.key] = InstrumentState(book)
        state.features = x
        self.feature_updates += 1

        if self.recorder is not None:
            self.recorder.append(book.last_update_ns, book, x)

        if self.predictor is not None:
            pred = self.predictor.predict_proba(x)
            self.infer_us.append(pred.latency_us)
            state.prediction = pred
            if pred.signal != state.signal:
                state.signal = pred.signal
                if pred.signal != "FLAT":
                    p = pred.p_up if pred.signal == "LONG" else pred.p_down
                    self.signals.appendleft(SignalEvent(book.last_update_ns, book.security_id, pred.signal, p))
                    logger.info(
                        "%s %d  p=%.3f  mid=%.2f", pred.signal, book.security_id, p, x[F["mid_price"]]
                    )

        for sink in tuple(self.sinks):
            sink.on_features(book, x)

        self.e2e_us.append((time.time_ns() - book.last_update_ns) / 1e3)


class EventLog(logging.Handler):
    """Keeps recent log lines for the dashboard, since stderr is hidden behind it."""

    def __init__(self, maxlen: int = 6) -> None:
        super().__init__()
        self.lines: deque[tuple[int, str]] = deque(maxlen=maxlen)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append((record.levelno, self.format(record)))


def _pct(values: deque[float]) -> str:
    if not values:
        return "-"
    p50, p99 = np.percentile(np.fromiter(values, dtype=np.float64), [50, 99])
    return f"{p50:,.0f} / {p99:,.0f}"


class Dashboard:
    def __init__(
        self,
        engine: Engine,
        stats: FeedStats,
        focus: BookKey,
        focus_label: str,
        events: EventLog,
        mode: str,
        portal_url: str | None = None,
    ) -> None:
        self.portal_url = portal_url
        self.engine = engine
        self.stats = stats
        self.focus = focus
        self.focus_label = focus_label
        self.events = events
        self.mode = mode
        self._last_rate = (time.monotonic(), 0)
        self._rate = 0.0

    def render(self) -> RenderableType:
        state = self.engine.states.get(self.focus)
        book = self.engine.books.get(self.focus)
        body = Table.grid(expand=True, padding=(0, 1))
        body.add_column(ratio=1)
        body.add_column(ratio=1)
        body.add_row(self._ladder(book), self._metrics(state))
        return Group(
            self._header(),
            body,
            self._prediction(state),
            self._footer(),
            self._events(),
        )

    def _header(self) -> RenderableType:
        now = time.monotonic()
        then, count = self._last_rate
        if now - then >= 1.0:
            self._rate = (self.stats.packets - count) / (now - then)
            self._last_rate = (now, self.stats.packets)
        if self.stats.connected:
            age = (time.time_ns() - self.stats.last_frame_ns) / 1e9 if self.stats.last_frame_ns else None
            status = Text("● LIVE", style="bold green") if age is not None and age < 5 else Text(
                "● CONNECTED (idle)", style="bold yellow"
            )
        else:
            status = Text("● RECONNECTING", style="bold red")
        clock = datetime.now(IST).strftime("%H:%M:%S IST")
        title = Text.assemble(
            ("AETHER ENGINE  ", "bold cyan"), (self.focus_label, "bold"), "  ", status,
            f"  {self.mode}  {clock}",
            (f"  lab {self.portal_url}" if self.portal_url else "", "cyan"),
        )
        return Panel(title, box=box.HEAVY)

    def _ladder(self, book: OrderBook | None) -> RenderableType:
        table = Table(box=box.SIMPLE_HEAD, expand=True, title=f"Top {LADDER_LEVELS} Depth")
        for name, style in (("Orders", "dim"), ("Bid Qty", "green"), ("Bid", "bold green"),
                            ("Ask", "bold red"), ("Ask Qty", "red"), ("Orders", "dim")):
            table.add_column(name, justify="right", style=style)
        for i in range(LADDER_LEVELS):
            if book is None:
                table.add_row(*["-"] * 6)
                continue
            bid = (f"{book.bid_orders[i]:,}", f"{book.bid_qty[i]:,}", f"{book.bid_px[i]:,.2f}") \
                if book.bid_px[i] > 0 else ("", "", "")
            ask = (f"{book.ask_px[i]:,.2f}", f"{book.ask_qty[i]:,}", f"{book.ask_orders[i]:,}") \
                if book.ask_px[i] > 0 else ("", "", "")
            table.add_row(*bid, *ask)
        return Panel(table, box=box.ROUNDED)

    def _metrics(self, state: InstrumentState | None) -> RenderableType:
        if state is None or state.features is None:
            return Panel(Text("Waiting for a two-sided book...", style="dim"), title="Microstructure")
        x = state.features
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column(justify="right")
        grid.add_row("Mid", f"{x[F['mid_price']]:,.2f}")
        grid.add_row("Micro-price", f"{x[F['micro_price']]:,.3f}  ({x[F['micro_offset_bps']]:+.2f} bps)")
        grid.add_row("Spread", f"{x[F['spread']]:,.2f}  ({x[F['spread_bps']]:.2f} bps)")
        grid.add_row("OBI L1 / 5 / 20", " / ".join(_obi(x[F[n]]) for n in ("obi_l1", "obi_5", "obi_20")))

        spectrum = Table(box=box.SIMPLE, title="Spectrum (volume by distance from mid)", expand=True)
        for col in ("Band", "Bid vol", "Ask vol", "Imbalance"):
            spectrum.add_column(col, justify="right")
        for i, label in enumerate(("≤0.05%", "0.05–0.15%", "0.15–0.50%"), start=1):
            spectrum.add_row(
                label,
                f"{x[F[f'bid_band_{i}']]:,.0f}",
                f"{x[F[f'ask_band_{i}']]:,.0f}",
                _obi(x[F[f'band_{i}_imbalance']]),
            )
        return Panel(Group(grid, spectrum), title="Microstructure", box=box.ROUNDED)

    def _prediction(self, state: InstrumentState | None) -> RenderableType:
        if self.engine.predictor is None:
            return Panel(Text("No model loaded - run with --model model.txt", style="dim"), title="Model")
        pred = state.prediction if state else None
        if pred is None:
            return Panel(Text("Waiting for features...", style="dim"), title="Model")
        threshold = self.engine.predictor.threshold
        bars = Table.grid(padding=(0, 1))
        for label, p, style in (("P(UP)  ", pred.p_up, "green"), ("P(FLAT)", pred.p_flat, "white"),
                                ("P(DOWN)", pred.p_down, "red")):
            filled = int(round(p * 30))
            bars.add_row(label, Text("█" * filled + "░" * (30 - filled), style=style), f"{p:.3f}")
        signal_style = {"LONG": "bold black on green", "SHORT": "bold white on red", "FLAT": "bold white on grey30"}
        sig = Text(f"  {pred.signal}  ", style=signal_style[pred.signal])

        history = Table.grid(padding=(0, 1))
        for ev in self.engine.signals:
            ts = datetime.fromtimestamp(ev.recv_ns / 1e9, IST).strftime("%H:%M:%S.%f")[:-3]
            history.add_row(ts, Text(ev.signal, style="green" if ev.signal == "LONG" else "red"),
                            str(ev.security_id), f"{ev.probability:.3f}")
        grid = Table.grid(expand=True, padding=(0, 3))
        grid.add_column()
        grid.add_column(justify="center", vertical="middle")
        grid.add_column()
        grid.add_row(bars, Group(Text("SIGNAL", style="dim"), sig), history)
        return Panel(grid, title=f"LightGBM  (threshold {threshold:.2f})", box=box.ROUNDED)

    def _footer(self) -> RenderableType:
        e, s = self.engine, self.stats
        rec = e.recorder
        parts = [
            f"pkts/s {self._rate:,.0f}",
            f"frames {s.frames:,}",
            f"dropped {s.dropped:,}",
            f"connects {s.connects}",
            f"skipped books {e.untradeable:,}",
            f"features µs p50/p99 {_pct(e.calc_us)}",
            f"infer µs {_pct(e.infer_us)}",
            f"tick→signal µs {_pct(e.e2e_us)}",
        ]
        if rec is not None:
            parts.append(f"recorded {rec.rows_written:,} (+{rec.pending:,} buffered)")
        return Panel(Text("   ".join(parts), style="dim"), box=box.ROUNDED)

    def _events(self) -> RenderableType:
        styles = {logging.WARNING: "yellow", logging.ERROR: "red", logging.CRITICAL: "bold red"}
        lines = [Text(line, style=styles.get(level, "")) for level, line in self.events.lines]
        return Panel(Group(*lines) if lines else Text("-", style="dim"), title="Events", box=box.ROUNDED)


def _obi(v: float) -> str:
    color = "green" if v > 0.2 else "red" if v < -0.2 else "white"
    return f"[{color}]{v:+.2f}[/]"


async def _flusher(recorder: ParquetRecorder, interval_s: float) -> None:
    last = time.monotonic()
    while True:
        await asyncio.sleep(1.0)
        if recorder.should_flush or (recorder.pending and time.monotonic() - last >= interval_s):
            await recorder.flush()
            last = time.monotonic()


async def _dashboard(dash: Dashboard, console: Console, refresh_hz: float) -> None:
    with Live(dash.render(), console=console, screen=True, auto_refresh=False) as live:
        while True:
            await asyncio.sleep(1.0 / refresh_hz)
            live.update(dash.render(), refresh=True)


async def _status_logger(engine: Engine, stats: FeedStats, interval_s: float = 10.0) -> None:
    while True:
        await asyncio.sleep(interval_s)
        rec = engine.recorder
        logger.info(
            "connected=%s packets=%d dropped=%d features=%d skipped=%d e2e_us(p50/p99)=%s%s",
            stats.connected, stats.packets, stats.dropped, engine.feature_updates, engine.untradeable,
            _pct(engine.e2e_us), f" recorded={rec.rows_written + rec.pending}" if rec else "",
        )


def _resolve_focus(arg: str | None, config: FeedConfig) -> tuple[BookKey, str]:
    if arg:
        segment, _, security_id = arg.partition(":")
    else:
        segment, security_id = config.instruments[0].exchange_segment, config.instruments[0].security_id
    return (ExchangeSegment[segment].value, int(security_id)), f"{segment}:{security_id}"


async def run(args: argparse.Namespace, console: Console, events: EventLog) -> None:
    if args.no_feed and not args.portal:
        raise ValueError("--no-feed only makes sense with --portal (a replay-only lab)")

    # Bind first so a busy port fails before anything connects to Dhan.
    sock = bind(args.host, args.port) if args.portal else None
    portal_url = f"http://{args.host}:{args.port}" if sock else None

    config: FeedConfig | None = None
    client: DepthFeedClient | None = None
    engine: Engine | None = None
    recorder: ParquetRecorder | None = None
    feed: asyncio.Task | None = None
    workers: list[asyncio.Task] = []

    if not args.no_feed:
        config = FeedConfig.from_env()
        queue: asyncio.Queue[FeedMessage] = asyncio.Queue(maxsize=args.queue_size)
        client = DepthFeedClient(config, queue)
        predictor = SignalPredictor(args.model, args.threshold) if args.model else None
        recorder = ParquetRecorder(args.data_dir) if args.record else None
        engine = Engine(queue, predictor, recorder)
        focus, focus_label = _resolve_focus(args.focus, config)
        mode = " + ".join(
            m for m, on in (("RECORD", recorder), ("PREDICT", predictor), ("LAB", sock)) if on
        ) or "VIEW"
        logger.info("Starting in %s mode for %d instrument(s)", mode, len(config.instruments))

        feed = asyncio.create_task(client.run(), name="feed")
        workers.append(asyncio.create_task(engine.run(), name="engine"))
        if recorder is not None:
            workers.append(asyncio.create_task(_flusher(recorder, args.flush_interval), name="flusher"))
        if args.headless:
            workers.append(asyncio.create_task(_status_logger(engine, client.stats), name="status"))
        else:
            dash = Dashboard(engine, client.stats, focus, focus_label, events, mode, portal_url)
            workers.append(asyncio.create_task(_dashboard(dash, console, args.refresh_hz), name="dashboard"))

    manager: ExperimentManager | None = None
    server = server_task = None
    if sock is not None:
        manager = ExperimentManager(
            ExperimentStore(args.experiments_dir),
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            engine=engine,
            instruments=config.instruments if config else (),
            feed_stats=client.stats if client else None,
        )
        server = make_server(create_app(manager))
        server_task = asyncio.create_task(server.serve(sockets=[sock]), name="portal")
        workers.append(asyncio.create_task(manager.autosave(), name="autosave"))
        logger.info("Paper trading lab at %s", portal_url)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows: KeyboardInterrupt still cancels run()
            pass
    tasks = [t for t in (feed, server_task, *workers) if t is not None]
    for task in tasks:
        task.add_done_callback(lambda _: stop.set())  # any task ending brings the system down

    try:
        await stop.wait()
    finally:
        if manager is not None:
            # Square off live experiments while the book is still current, and save everything.
            await manager.shutdown()
        if client is not None:
            await client.stop()
            try:
                await asyncio.wait_for(asyncio.shield(feed), timeout=5)
            except Exception:  # timeout, or the feed already failed (re-raised below)
                feed.cancel()
        if engine is not None:
            engine.drain()
        if server is not None:
            server.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(server_task), timeout=5)
            except Exception:
                server_task.cancel()
        for task in workers:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if recorder is not None:
            recorder.close()
            logger.info("Recorder closed: %d rows in %d files", recorder.rows_written, recorder.files_written)

    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true", help="write feature vectors to Parquet")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--flush-interval", type=float, default=60.0, help="max seconds between Parquet flushes")
    parser.add_argument("--model", type=Path, default=None, help="LightGBM model.txt for live signals")
    parser.add_argument("--threshold", type=float, default=0.70, help="probability needed for LONG/SHORT")
    parser.add_argument("--focus", default=None, help="SEGMENT:SECURITY_ID shown on the dashboard")
    parser.add_argument("--headless", action="store_true", help="no dashboard; log status lines instead")
    parser.add_argument("--refresh-hz", type=float, default=4.0,
                        help="dashboard refresh rate; rendering shares the event loop with the feed")
    parser.add_argument("--queue-size", type=int, default=10_000)
    parser.add_argument("--log-file", type=Path, default=None)
    lab = parser.add_argument_group("paper trading lab")
    lab.add_argument("--portal", action="store_true", help="serve the paper trading lab web portal")
    lab.add_argument("--host", default="127.0.0.1", help="portal address; keep it local, the process holds your token")
    lab.add_argument("--port", type=int, default=8000)
    lab.add_argument("--no-feed", action="store_true", help="portal only, for replay experiments without Dhan")
    lab.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    lab.add_argument("--models-dir", type=Path, default=Path("."), help="where the portal looks for model files")
    args = parser.parse_args()
    if args.no_feed:
        args.headless = True  # nothing for the terminal dashboard to show

    console = Console()
    events = EventLog()
    handlers: list[logging.Handler] = [logging.StreamHandler()] if args.headless else [events]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in handlers:
        if h is not events:
            h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers)

    try:
        asyncio.run(run(args, console, events))
    except FeedFatalError as exc:
        console.print(f"[bold red]Feed rejected the session:[/] {exc}")
        sys.exit(1)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]Configuration error:[/] {exc}")
        sys.exit(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
