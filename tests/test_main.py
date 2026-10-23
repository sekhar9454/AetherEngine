"""End-to-end: engine and dashboard in-process, and the real CLI as a subprocess."""

import asyncio
import os
import signal
import sys
from pathlib import Path

import polars as pl
import pytest
from rich.console import Console

import src.feed.websocket_client as wc
from src.inference.predictor import SignalPredictor
from src.main import Dashboard, Engine, EventLog
from src.pipeline.recorder import ParquetRecorder
from tests.conftest import mock_dhan

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_engine_and_dashboard_against_mock_feed(model_path, tmp_path, monkeypatch):
    async def scenario():
        async with mock_dhan() as (url, received):
            monkeypatch.setattr(wc, "DEPTH_ENDPOINT", url)
            queue = asyncio.Queue(maxsize=10_000)
            client = wc.DepthFeedClient(wc.FeedConfig("1000", "tok", (wc.Instrument("NSE_EQ", "1333"),)), queue)
            engine = Engine(queue, SignalPredictor(model_path), ParquetRecorder(tmp_path))
            tasks = [asyncio.create_task(client.run()), asyncio.create_task(engine.run())]
            await asyncio.sleep(2.0)
            dash = Dashboard(engine, client.stats, (1, 1333), "NSE_EQ:1333", EventLog(), "RECORD + PREDICT")
            console = Console(record=True, width=150, force_terminal=False)
            console.print(dash.render())
            await client.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            engine.recorder.close()
            return engine, client, console.export_text(), received

    engine, client, screen, received = asyncio.run(scenario())
    assert received[0]["RequestCode"] == 23
    assert client.stats.dropped == 0 and engine.untradeable == 0
    assert engine.feature_updates > 50
    assert engine.states[(1, 1333)].prediction is not None
    assert engine.recorder.rows_written == engine.feature_updates
    for text in ("AETHER ENGINE", "Top 5 Depth", "Microstructure", "P(UP)", "SIGNAL", "tick→signal"):
        assert text in screen


def _env(url: str) -> dict[str, str]:
    return dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        TEST_DEPTH_ENDPOINT=url,
        DHAN_CLIENT_ID="1000",
        DHAN_ACCESS_TOKEN="tok",
        DHAN_INSTRUMENTS="NSE_EQ:1333",
    )


@pytest.mark.slow
def test_cli_records_and_shuts_down_cleanly_on_sigint(model_path, tmp_path):
    data = tmp_path / "data"

    async def scenario():
        async with mock_dhan() as (url, received):
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(ROOT / "tests" / "_run_main.py"),
                "--record", "--data-dir", str(data), "--model", str(model_path), "--headless",
                cwd=tmp_path, env=_env(url),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            await asyncio.sleep(4.0)
            proc.send_signal(signal.SIGINT)
            out, _ = await asyncio.wait_for(proc.communicate(), 15)
            await asyncio.sleep(0.1)
            return proc.returncode, out.decode(), received

    code, out, received = asyncio.run(scenario())
    assert code == 0, out
    assert "Recorder closed" in out
    assert [m["RequestCode"] for m in received] == [23, 12]  # subscribe, then graceful disconnect
    rows = pl.read_parquet(sorted(data.rglob("*.parquet")))
    assert len(rows) > 50 and rows["recv_ns"].is_sorted()


@pytest.mark.slow
def test_cli_exits_with_error_on_auth_failure(tmp_path):
    async def scenario():
        async with mock_dhan(mode="fatal") as (url, _):
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(ROOT / "tests" / "_run_main.py"), "--headless",
                cwd=tmp_path, env=_env(url),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), 15)
            return proc.returncode, out.decode()

    code, out = asyncio.run(scenario())
    assert code == 1
    assert "808" in out and "Feed rejected the session" in out
