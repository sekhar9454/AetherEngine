import asyncio
import struct

from src.feed.binary_parser import DEPTH_PACKET_SIZE, parse_frame, push_latest
from tests.synth import depth_packet, disconnect_packet

STACKED = depth_packet(41, 1333, 1, 1500.0) + depth_packet(51, 1333, 2, 1500.5) + depth_packet(41, 11536, 3, 4000.0)


def test_packet_size():
    assert len(depth_packet(41, 1, 1, 1.0)) == DEPTH_PACKET_SIZE == 332


def test_stacked_packets_are_split():
    msgs = parse_frame(STACKED, recv_ns=123)
    assert [m["side"] for m in msgs] == ["bid", "ask", "bid"]
    assert [m["security_id"] for m in msgs] == [1333, 1333, 11536]
    assert all(m["recv_ns"] == 123 and m["exchange_segment"] == 1 for m in msgs)


def test_levels_decoded():
    m = parse_frame(STACKED)[0]
    assert len(m["prices"]) == len(m["quantities"]) == len(m["orders"]) == 20
    assert m["prices"][0] == 1500.0 and abs(m["prices"][19] - 1500.95) < 1e-9
    assert m["quantities"][:3] == [100, 101, 102]
    assert m["orders"][:3] == [1, 2, 3]


def test_disconnect_packet():
    (d,) = parse_frame(disconnect_packet(807))
    assert d["kind"] == "disconnect" and d["reason_code"] == 807 and "expired" in d["reason"]


def test_truncated_tail_dropped():
    assert len(parse_frame(STACKED[:-10])) == 2


def test_unknown_packet_skipped_by_length():
    unknown = struct.pack("<hBBII", 20, 99, 1, 1, 1) + b"\x00" * 8
    (m,) = parse_frame(unknown + depth_packet(51, 7, 1, 10.0))
    assert m["security_id"] == 7 and m["side"] == "ask"


def test_unknown_packet_with_bad_length_stops_parsing():
    assert parse_frame(struct.pack("<hBBII", 0, 99, 1, 1, 1) + b"junk") == []


def test_empty_and_short_frames():
    assert parse_frame(b"") == []
    assert parse_frame(b"\x01\x02") == []


def test_push_latest_evicts_oldest():
    q: asyncio.Queue[int] = asyncio.Queue(maxsize=2)
    assert push_latest(q, 1) and push_latest(q, 2)
    assert not push_latest(q, 3)
    assert [q.get_nowait(), q.get_nowait()] == [2, 3]
