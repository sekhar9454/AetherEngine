"""Shared fixtures: synthetic recorded sessions, a trained model, and a mock Dhan server."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
from collections.abc import AsyncIterator
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console
from websockets.asyncio.server import serve

from src.features.calculator import compute_features
from src.features.orderbook import BookRegistry
from src.pipeline.recorder import IST, ParquetRecorder
from src.pipeline.trainer import TrainConfig, train
from tests.synth import SynthMarket, disconnect_packet

SESSIONS = (date(2026, 10, 20), date(2026, 10, 21))


def record_session(recorder: ParquetRecorder, day: date, seed: int, mean_step_s: float = 0.6) -> None:
    """Feed one 09:15-15:30 IST session of synthetic updates through book -> features -> recorder."""
    market, books, rng = SynthMarket(seed), BookRegistry(), np.random.default_rng(seed)
    ns = int(datetime.combine(day, time(9, 15), tzinfo=IST).timestamp() * 1e9)
    end_ns = int(datetime.combine(day, time(15, 30), tzinfo=IST).timestamp() * 1e9)
    while ns < end_ns:
        for update in market.updates(ns):
            book = books.apply(update)
        features = compute_features(book)
        if features is not None:
            recorder.append(ns, book, features)
        if recorder.should_flush:
            recorder.close()
        ns += int((rng.exponential(mean_step_s) + 0.001) * 1e9)
    recorder.close()


@pytest.fixture(scope="session")
def recorded_dir(tmp_path_factory) -> Path:
    """Two synthetic trading sessions recorded to Parquet."""
    data = tmp_path_factory.mktemp("data")
    recorder = ParquetRecorder(data, flush_rows=20_000)
    for seed, day in enumerate(SESSIONS, start=1):
        record_session(recorder, day, seed)
    return data


@pytest.fixture(scope="session")
def model_path(recorded_dir, tmp_path_factory) -> Path:
    """A model trained by the real trainer on the recorded sessions."""
    path = tmp_path_factory.mktemp("model") / "model.txt"
    train(TrainConfig(data_dir=recorded_dir, model_path=path), Console(file=io.StringIO()))
    return path


@contextlib.asynccontextmanager
async def mock_dhan(mode: str = "stream", hz: float = 100.0) -> AsyncIterator[tuple[str, list]]:
    """A local Dhan 20-depth server. Yields (url, received client messages).

    mode "stream" sends SynthMarket frames at ``hz``; "fatal" sends disconnect 808.
    """
    received: list[dict] = []

    async def handler(ws):
        received.append(json.loads(await ws.recv()))
        if mode == "fatal":
            await ws.send(disconnect_packet(808))
            await ws.wait_closed()
            return
        market = SynthMarket(7)

        async def reader():
            async for msg in ws:
                received.append(json.loads(msg))

        reading = asyncio.create_task(reader())
        try:
            while True:
                await ws.send(market.frame())
                await asyncio.sleep(1 / hz)
        except Exception:
            pass
        finally:
            reading.cancel()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/twentydepth", received
