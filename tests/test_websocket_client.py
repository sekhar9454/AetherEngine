"""Client behaviour against a local mock of the Dhan 20-depth server."""

import asyncio
import json

import pytest
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

import src.feed.websocket_client as wc
from tests.synth import depth_packet, disconnect_packet

FRAME = depth_packet(41, 1333, 1, 1500.0) + depth_packet(51, 1333, 2, 1500.5) + depth_packet(41, 11536, 3, 4000.0)
ONE = (wc.Instrument("NSE_EQ", "1"),)


def _port(server) -> int:
    return server.sockets[0].getsockname()[1]


@pytest.fixture
def endpoint(monkeypatch):
    def point_at(url: str) -> None:
        monkeypatch.setattr(wc, "DEPTH_ENDPOINT", url)
    return point_at


def test_config_validation():
    for bad in (
        lambda: wc.Instrument("NSE_XX", "1"),
        lambda: wc.Instrument("NSE_EQ", "abc"),
        lambda: wc.FeedConfig("", "t", ONE),
        lambda: wc.FeedConfig("1", "t", ()),
    ):
        with pytest.raises(ValueError):
            bad()


def test_token_not_in_repr():
    assert "secret-token" not in repr(wc.FeedConfig("1", "secret-token", ONE))


def test_reconnect_resubscribe_then_fatal_disconnect(endpoint):
    paths, subs = [], []

    async def handler(ws):
        paths.append(ws.request.path)
        subs.append(json.loads(await ws.recv()))
        if len(paths) == 1:
            await ws.send(FRAME)
            ws.transport.abort()  # abrupt network drop, no close frame
        else:
            await ws.send(depth_packet(51, 1333, 9, 1501.0) + disconnect_packet(807))
            await ws.wait_closed()

    async def scenario():
        async with serve(handler, "127.0.0.1", 0) as server:
            endpoint(f"ws://127.0.0.1:{_port(server)}/twentydepth")
            cfg = wc.FeedConfig(
                "1000", "tok", (wc.Instrument("NSE_EQ", "1333"), wc.Instrument("NSE_EQ", "11536")),
                initial_backoff_s=0.05,
            )
            queue = asyncio.Queue()
            client = wc.DepthFeedClient(cfg, queue)
            with pytest.raises(wc.FeedFatalError, match="807"):
                await asyncio.wait_for(client.run(), 5)
            return client, queue

    client, queue = asyncio.run(scenario())
    assert len(paths) == 2
    assert all(p in paths[0] for p in ("token=tok", "clientId=1000", "authType=2"))
    assert subs[0] == subs[1]
    assert subs[0]["RequestCode"] == 23 and subs[0]["InstrumentCount"] == 2
    assert [queue.get_nowait()["kind"] for _ in range(queue.qsize())] == ["depth"] * 4 + ["disconnect"]
    assert client.stats.connects == 2 and client.stats.packets == 5


def test_server_error_retries_with_backoff(endpoint):
    attempts = 0

    async def handler(ws):
        nonlocal attempts
        attempts += 1
        await ws.recv()
        await ws.send(disconnect_packet(800))  # non-fatal
        await ws.close()

    async def scenario():
        async with serve(handler, "127.0.0.1", 0) as server:
            endpoint(f"ws://127.0.0.1:{_port(server)}/")
            client = wc.DepthFeedClient(wc.FeedConfig("1", "t", ONE, initial_backoff_s=0.02), asyncio.Queue())
            task = asyncio.create_task(client.run())
            await asyncio.sleep(0.5)
            await client.stop()
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
    # Exponential backoff: a handful of attempts in 0.5 s, not one per round trip.
    assert 3 <= attempts <= 8


def test_http_401_is_fatal(endpoint):
    def reject(connection, request):
        return Response(401, "Unauthorized", Headers(), b"bad token")

    async def scenario():
        async with serve(lambda ws: None, "127.0.0.1", 0, process_request=reject) as server:
            endpoint(f"ws://127.0.0.1:{_port(server)}/")
            client = wc.DepthFeedClient(wc.FeedConfig("1", "t", ONE), asyncio.Queue())
            with pytest.raises(wc.FeedFatalError, match="401"):
                await asyncio.wait_for(client.run(), 3)

    asyncio.run(scenario())


def test_graceful_stop_sends_disconnect_request(endpoint):
    got = []

    async def handler(ws):
        async for msg in ws:
            got.append(json.loads(msg))
            if got[-1]["RequestCode"] == 23:
                await ws.send(FRAME)

    async def scenario():
        async with serve(handler, "127.0.0.1", 0) as server:
            endpoint(f"ws://127.0.0.1:{_port(server)}/")
            queue = asyncio.Queue()
            client = wc.DepthFeedClient(wc.FeedConfig("1", "t", ONE), queue)
            task = asyncio.create_task(client.run())
            await asyncio.wait_for(queue.get(), 2)
            await client.stop()
            await asyncio.wait_for(task, 3)
            await asyncio.sleep(0.05)
            return client

    client = asyncio.run(scenario())
    assert [g["RequestCode"] for g in got] == [23, 12]
    assert client.stats.connects == 1


def test_connection_refused_keeps_retrying(endpoint):
    async def scenario():
        endpoint("ws://127.0.0.1:1/")
        client = wc.DepthFeedClient(wc.FeedConfig("1", "t", ONE, initial_backoff_s=0.02), asyncio.Queue())
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.3)
        assert not task.done()
        await client.stop()
        await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
