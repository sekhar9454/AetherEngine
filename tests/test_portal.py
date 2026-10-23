"""Portal API and experiment manager: replay on recorded data, live on the mock feed."""

import asyncio
import csv
import io
import os
import shutil
import signal
import sys
from pathlib import Path

import httpx
import pytest

import src.feed.websocket_client as wc
from src.main import Engine
from src.paper.manager import ExperimentManager, ExperimentStore
from src.portal.server import bind, create_app
from tests.conftest import SESSIONS, mock_dhan

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


@pytest.fixture
def models_dir(model_path, tmp_path) -> Path:
    folder = tmp_path / "models"
    folder.mkdir()
    shutil.copy(model_path, folder / "model.txt")
    (folder / "requirements.txt").write_text("numpy\n")  # not a model: must not be listed
    return folder


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://lab")


async def _wait_done(http: httpx.AsyncClient, exp_id: str, timeout: float = 30) -> dict:
    for _ in range(int(timeout / 0.1)):
        detail = (await http.get(f"/api/experiments/{exp_id}")).json()
        if detail["status"] != "running":
            return detail
        await asyncio.sleep(0.1)
    raise AssertionError("experiment did not finish")


REPLAY = {
    "mode": "replay", "security_id": 1333, "model": "model.txt", "days": [SESSIONS[0].isoformat()],
    "starting_capital": 200_000, "trade_size": 50_000, "hold_seconds": 3, "latency_ms": 100,
}


def test_replay_experiment_end_to_end(recorded_dir, models_dir, tmp_path):
    store_dir = tmp_path / "experiments"

    async def scenario():
        manager = ExperimentManager(ExperimentStore(store_dir), recorded_dir, models_dir)
        async with _client(create_app(manager)) as http:
            meta = (await http.get("/api/meta")).json()
            assert meta["models"] == ["model.txt"]
            assert {s["date"] for s in meta["sessions"]} == {d.isoformat() for d in SESSIONS}
            assert meta["live"]["available"] is False
            assert meta["defaults"]["signal_threshold"] == 0.7

            r = await http.post("/api/experiments", json=REPLAY | {"name": "baseline"})
            assert r.status_code == 201, r.text
            detail = await _wait_done(http, r.json()["id"])
            listing = (await http.get("/api/experiments")).json()
            csv_text = (await http.get(f"/api/experiments/{detail['id']}/trades.csv")).text
            return detail, listing, csv_text

    detail, listing, csv_text = asyncio.run(scenario())
    m = detail["metrics"]
    assert detail["status"] == "completed" and detail["progress"] == 1.0
    assert detail["name"] == "baseline" and listing[0]["id"] == detail["id"]
    assert m["trades"] > 10 and m["charges"] > 0
    assert m["net_pnl"] == pytest.approx(m["gross_pnl"] - m["charges"], abs=0.01)
    assert m["final_equity"] == pytest.approx(200_000 + m["net_pnl"])
    assert [d["date"] for d in detail["days"]] == [SESSIONS[0].isoformat()]
    assert len(detail["equity"]) > 10 and detail["config"]["latency_ms"] == 100
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == m["trades"]
    assert sum(float(r["net_pnl"]) for r in rows) == pytest.approx(m["net_pnl"], abs=0.05 * m["trades"])

    # Saved to disk and reloaded by a fresh manager (e.g. after a restart).
    reloaded = ExperimentManager(ExperimentStore(store_dir), recorded_dir, models_dir)
    assert reloaded.get(detail["id"])["metrics"] == m


def test_replay_validation_errors(recorded_dir, models_dir, tmp_path):
    async def scenario():
        manager = ExperimentManager(ExperimentStore(tmp_path / "e"), recorded_dir, models_dir)
        async with _client(create_app(manager)) as http:
            return [
                (await http.post("/api/experiments", json=REPLAY | body)).status_code
                for body in (
                    {"model": "requirements.txt"},     # not a model
                    {"days": ["2020-01-01"]},           # nothing recorded
                    {"security_id": 999},               # unknown instrument
                    {"signal_threshold": 0.3},          # pydantic range
                    {"mode": "live"},                   # no feed in this lab
                )
            ]

    assert asyncio.run(scenario()) == [400, 400, 400, 422, 400]


def test_stop_and_delete(recorded_dir, models_dir, tmp_path):
    async def scenario():
        manager = ExperimentManager(ExperimentStore(tmp_path / "e"), recorded_dir, models_dir)
        async with _client(create_app(manager)) as http:
            exp_id = (await http.post("/api/experiments", json=REPLAY | {"days": []})).json()["id"]
            in_flight_delete = (await http.delete(f"/api/experiments/{exp_id}")).status_code
            stop = (await http.post(f"/api/experiments/{exp_id}/stop")).status_code
            detail = await _wait_done(http, exp_id)
            delete = (await http.delete(f"/api/experiments/{exp_id}")).status_code
            gone = (await http.get(f"/api/experiments/{exp_id}")).status_code
            return in_flight_delete, stop, detail["status"], delete, gone

    in_flight_delete, stop, status, delete, gone = asyncio.run(scenario())
    assert in_flight_delete == 409 and stop == 200
    assert status in ("stopped", "completed")
    assert delete == 200 and gone == 404
    assert not list((tmp_path / "e").iterdir())


def test_live_experiment_on_mock_feed(models_dir, tmp_path, monkeypatch):
    async def scenario():
        async with mock_dhan(hz=100) as (url, received):
            monkeypatch.setattr(wc, "DEPTH_ENDPOINT", url)
            instruments = (wc.Instrument("NSE_EQ", "1333"),)
            queue = asyncio.Queue(maxsize=10_000)
            client = wc.DepthFeedClient(wc.FeedConfig("1000", "tok", instruments), queue)
            engine = Engine(queue, None, None)
            tasks = [asyncio.create_task(client.run()), asyncio.create_task(engine.run())]
            manager = ExperimentManager(
                ExperimentStore(tmp_path / "e"), tmp_path / "no-data", models_dir,
                engine=engine, instruments=instruments, feed_stats=client.stats,
            )
            async with _client(create_app(manager)) as http:
                bad = await http.post("/api/experiments", json=REPLAY | {"mode": "live", "security_id": 42})
                r = await http.post("/api/experiments", json=REPLAY | {"mode": "live", "days": [],
                                                                       "start_time": "00:00",
                                                                       "square_off_time": "23:59"})
                exp_id = r.json()["id"]
                await asyncio.sleep(2.5)
                running = (await http.get(f"/api/experiments/{exp_id}")).json()
                await http.post(f"/api/experiments/{exp_id}/stop")
                final = (await http.get(f"/api/experiments/{exp_id}")).json()
            await client.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return bad, running, final, engine

    bad, running, final, engine = asyncio.run(scenario())
    assert bad.status_code == 400 and "not subscribed" in bad.text
    assert running["status"] == "running" and running["metrics"]["ticks"] > 50
    assert running["prediction"]["signal"] in ("LONG", "SHORT", "FLAT")
    assert final["status"] == "stopped" and final["position"] is None
    assert final["metrics"]["final_equity"] == pytest.approx(200_000 + final["metrics"]["net_pnl"])
    assert engine.sinks == []  # detached on stop
    assert (tmp_path / "e" / final["id"] / "experiment.json").exists()


def test_portal_cli_serves_and_shuts_down(recorded_dir, models_dir, tmp_path):
    """The real CLI: --portal --no-feed starts uvicorn, answers HTTP, exits 0 on SIGINT."""
    sock = bind("127.0.0.1", 0)
    port = sock.getsockname()[1]
    sock.close()

    async def scenario():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "src.main", "--portal", "--no-feed", "--port", str(port),
            "--data-dir", str(recorded_dir), "--models-dir", str(models_dir),
            "--experiments-dir", str(tmp_path / "e"),
            cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
            for _ in range(100):
                try:
                    page = await http.get("/")
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.1)
            meta = (await http.get("/api/meta")).json()
        proc.send_signal(signal.SIGINT)
        out, _ = await asyncio.wait_for(proc.communicate(), 15)
        return page, meta, proc.returncode, out.decode()

    page, meta, code, out = asyncio.run(scenario())
    assert page.status_code == 200 and "Paper Trading Lab" in page.text
    assert meta["models"] == ["model.txt"]
    assert code == 0, out
    assert "Paper trading lab at" in out


def test_busy_port_is_a_clear_error(tmp_path):
    held = bind("127.0.0.1", 0)
    held.listen()
    port = held.getsockname()[1]
    try:
        with pytest.raises(ValueError, match="Cannot open the portal"):
            bind("127.0.0.1", port)
    finally:
        held.close()
