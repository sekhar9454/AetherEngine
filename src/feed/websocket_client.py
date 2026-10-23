"""Resilient asyncio client for the DhanHQ 20-level market depth WebSocket.

Authenticates via query string, subscribes with a JSON request, decodes binary
frames with ``binary_parser`` and pushes ``FeedMessage`` dicts onto a queue.
Transient failures reconnect with jittered exponential backoff and resubscribe;
credential/subscription errors stop the client with ``FeedFatalError``.

Smoke test against the live feed (reads .env):
    python -m src.feed.websocket_client
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Final
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from .binary_parser import (
    FATAL_DISCONNECT_CODES,
    ExchangeSegment,
    FeedMessage,
    parse_frame,
    push_latest,
)

logger = logging.getLogger(__name__)

DEPTH_ENDPOINT: Final = "wss://depth-api-feed.dhan.co/twentydepth"
SUBSCRIBE_TWENTY_DEPTH: Final = 23
DISCONNECT_REQUEST: Final = 12
MAX_INSTRUMENTS: Final = 50
# HTTP statuses on the handshake that retrying with the same credentials won't fix.
FATAL_HTTP_STATUSES: Final = frozenset({400, 401, 403})


class FeedFatalError(RuntimeError):
    """The server rejected the session in a way reconnecting cannot fix."""


@dataclass(frozen=True, slots=True)
class Instrument:
    exchange_segment: str  # e.g. "NSE_EQ"
    security_id: str

    def __post_init__(self) -> None:
        if self.exchange_segment not in ExchangeSegment.__members__:
            raise ValueError(f"Unknown exchange segment: {self.exchange_segment!r}")
        if not self.security_id.isdigit():
            raise ValueError(f"Security ID must be numeric: {self.security_id!r}")


@dataclass(frozen=True, slots=True)
class FeedConfig:
    client_id: str
    access_token: str = field(repr=False)
    instruments: tuple[Instrument, ...]
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 30.0
    open_timeout_s: float = 10.0
    # The server pings every 10 s; our own pings detect half-open TCP connections.
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0

    def __post_init__(self) -> None:
        if not self.client_id or not self.access_token:
            raise ValueError("client_id and access_token are required")
        if not 1 <= len(self.instruments) <= MAX_INSTRUMENTS:
            raise ValueError(f"Between 1 and {MAX_INSTRUMENTS} instruments per connection")

    @classmethod
    def from_env(cls) -> FeedConfig:
        """Build from DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN and DHAN_INSTRUMENTS (SEG:ID,...)."""
        from dotenv import load_dotenv

        load_dotenv()
        raw = os.environ.get("DHAN_INSTRUMENTS", "NSE_EQ:1333")
        instruments: list[Instrument] = []
        for pair in filter(None, (p.strip() for p in raw.split(","))):
            segment, sep, security_id = pair.partition(":")
            if not sep:
                raise ValueError(f"Expected SEGMENT:SECURITY_ID, got {pair!r}")
            instruments.append(Instrument(segment.strip(), security_id.strip()))
        return cls(
            client_id=os.environ.get("DHAN_CLIENT_ID", ""),
            access_token=os.environ.get("DHAN_ACCESS_TOKEN", ""),
            instruments=tuple(instruments),
        )

    @property
    def url(self) -> str:
        query = urlencode({"token": self.access_token, "clientId": self.client_id, "authType": 2})
        return f"{DEPTH_ENDPOINT}?{query}"


@dataclass(slots=True)
class FeedStats:
    connects: int = 0
    frames: int = 0
    packets: int = 0
    dropped: int = 0  # evicted from a full queue
    last_frame_ns: int = 0
    connected: bool = False


class DepthFeedClient:
    """Streams decoded depth packets into ``queue`` until ``stop()`` or a fatal error."""

    def __init__(self, config: FeedConfig, queue: asyncio.Queue[FeedMessage]) -> None:
        self._config = config
        self._queue = queue
        self._stopping = asyncio.Event()
        self._ws: ClientConnection | None = None
        self._fatal: FeedFatalError | None = None
        self.stats = FeedStats()

    async def run(self) -> None:
        backoff = self._config.initial_backoff_s
        while not self._stopping.is_set():
            got_depth = False
            try:
                async with connect(
                    self._config.url,
                    open_timeout=self._config.open_timeout_s,
                    ping_interval=self._config.ping_interval_s,
                    ping_timeout=self._config.ping_timeout_s,
                    close_timeout=2.0,
                    compression=None,
                    max_queue=None,
                ) as ws:
                    self._ws = ws
                    self.stats.connects += 1
                    self.stats.connected = True
                    logger.info("Connected to depth feed (attempt %d)", self.stats.connects)
                    await ws.send(self._subscription())
                    async for frame in ws:
                        got_depth = self._on_frame(frame) or got_depth
                        if self._fatal:
                            await ws.close()
                            break
                    else:
                        if not self._stopping.is_set():
                            logger.warning(
                                "Server closed connection: code=%s reason=%r",
                                ws.close_code, ws.close_reason,
                            )
            except InvalidStatus as exc:
                status = exc.response.status_code
                if status in FATAL_HTTP_STATUSES:
                    raise FeedFatalError(f"Handshake rejected with HTTP {status}") from exc
                logger.warning("Handshake failed with HTTP %d", status)
            except ConnectionClosed as exc:
                logger.warning("Connection lost: %s", exc)
            except (OSError, InvalidHandshake, TimeoutError) as exc:
                logger.warning("Connection failed: %s: %s", type(exc).__name__, exc)
            finally:
                self._ws = None
                self.stats.connected = False

            if self._fatal:
                raise self._fatal
            if self._stopping.is_set():
                break
            # Only real depth data proves the session is healthy; a server that
            # accepts and immediately errors must not reset the backoff.
            if got_depth:
                backoff = self._config.initial_backoff_s

            delay = random.uniform(0.5, 1.0) * backoff
            logger.info("Reconnecting in %.2fs", delay)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)
            except TimeoutError:
                pass
            backoff = min(backoff * 2, self._config.max_backoff_s)

    async def stop(self) -> None:
        """Unsubscribe gracefully and end ``run()``."""
        self._stopping.set()
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"RequestCode": DISCONNECT_REQUEST}))
        except ConnectionClosed:
            pass
        await ws.close()

    def _subscription(self) -> str:
        instruments = self._config.instruments
        return json.dumps(
            {
                "RequestCode": SUBSCRIBE_TWENTY_DEPTH,
                "InstrumentCount": len(instruments),
                "InstrumentList": [
                    {"ExchangeSegment": i.exchange_segment, "SecurityId": i.security_id}
                    for i in instruments
                ],
            }
        )

    def _on_frame(self, frame: str | bytes) -> bool:
        """Decode and enqueue a frame; returns True if it carried depth data."""
        recv_ns = time.time_ns()
        if isinstance(frame, str):
            logger.warning("Unexpected text frame: %.200s", frame)
            return False
        self.stats.frames += 1
        self.stats.last_frame_ns = recv_ns
        got_depth = False
        for message in parse_frame(frame, recv_ns):
            self.stats.packets += 1
            if not push_latest(self._queue, message):
                self.stats.dropped += 1
            if message["kind"] == "depth":
                got_depth = True
            else:
                code = message["reason_code"]
                logger.error("Server disconnect %d: %s", code, message["reason"])
                if code in FATAL_DISCONNECT_CODES:
                    self._fatal = FeedFatalError(f"Disconnect {code}: {message['reason']}")
        return got_depth


async def _smoke() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    queue: asyncio.Queue[FeedMessage] = asyncio.Queue(maxsize=10_000)
    client = DepthFeedClient(FeedConfig.from_env(), queue)

    async def consume() -> None:
        while True:
            msg: Any = await queue.get()
            if msg["kind"] == "depth":
                print(
                    f"{msg['security_id']:>8} {msg['side']:<3} seq={msg['sequence']:<10} "
                    f"L0 {msg['prices'][0]:>10.2f} x {msg['quantities'][0]:<8} "
                    f"({msg['orders'][0]} orders)"
                )
            else:
                print(f"DISCONNECT {msg['reason_code']}: {msg['reason']}")

    consumer = asyncio.create_task(consume())
    try:
        await client.run()
    finally:
        consumer.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(_smoke())
    except KeyboardInterrupt:
        pass
