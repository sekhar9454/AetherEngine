"""Decoder for DhanHQ 20-level market depth binary frames.

One WebSocket frame can carry several packets back to back (bid then ask for
each subscribed instrument), so frames are walked packet by packet.

Packet layout (little-endian):
    header   12 B   <hBBII    msg_len, response_code, exchange_segment, security_id, sequence
    depth   320 B   20 x <dII price, quantity, orders      (response code 41 = bid, 51 = ask)
    reason    2 B   <h        disconnect reason code       (response code 50)
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from enum import IntEnum
from typing import Final, Literal, TypeAlias, TypedDict, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEPTH_LEVELS: Final = 20

HEADER: Final = struct.Struct("<hBBII")
LEVEL: Final = struct.Struct("<dII")
# All 20 levels in one call yields a flat (price, qty, orders, price, ...) tuple,
# which is several times faster than iterating LEVEL.iter_unpack.
LEVELS: Final = struct.Struct("<" + "dII" * DEPTH_LEVELS)
DISCONNECT_REASON: Final = struct.Struct("<h")

DEPTH_PACKET_SIZE: Final = HEADER.size + LEVELS.size  # 332
DISCONNECT_PACKET_SIZE: Final = HEADER.size + DISCONNECT_REASON.size  # 14


class FeedCode(IntEnum):
    BID = 41
    DISCONNECT = 50
    ASK = 51


class ExchangeSegment(IntEnum):
    IDX_I = 0
    NSE_EQ = 1
    NSE_FNO = 2
    NSE_CURRENCY = 3
    BSE_EQ = 4
    MCX_COMM = 5
    BSE_CURRENCY = 7
    BSE_FNO = 8


DISCONNECT_REASONS: Final[dict[int, str]] = {
    800: "Internal server error",
    804: "Requested number of instruments exceeds limit",
    805: "Too many requests or connections",
    806: "Data APIs not subscribed",
    807: "Access token is expired",
    808: "Authentication failed - client ID or access token invalid",
    809: "Access token is invalid",
    810: "Client ID is invalid",
    811: "Invalid expiry date",
    812: "Invalid date format",
    813: "Invalid security ID",
    814: "Invalid request",
}

# Reconnecting cannot fix these; for 805, retrying risks the account being blocked.
FATAL_DISCONNECT_CODES: Final = frozenset(DISCONNECT_REASONS) - {800}


class DepthUpdate(TypedDict):
    kind: Literal["depth"]
    side: Literal["bid", "ask"]
    exchange_segment: int
    security_id: int
    sequence: int
    recv_ns: int  # wall-clock receive time; the feed carries no exchange timestamp
    prices: list[float]  # level 0 is best; empty levels are reported as price 0.0
    quantities: list[int]
    orders: list[int]


class DisconnectNotice(TypedDict):
    kind: Literal["disconnect"]
    exchange_segment: int
    security_id: int
    reason_code: int
    reason: str
    recv_ns: int


FeedMessage: TypeAlias = DepthUpdate | DisconnectNotice


def parse_frame(frame: bytes | bytearray | memoryview, recv_ns: int | None = None) -> list[FeedMessage]:
    """Decode every packet in a binary WebSocket frame.

    Truncated or unrecognised trailing data is logged and dropped rather than
    raised, so one bad packet never takes down the feed.
    """
    if recv_ns is None:
        recv_ns = time.time_ns()

    view = memoryview(frame)
    end = len(view)
    offset = 0
    messages: list[FeedMessage] = []

    while end - offset >= HEADER.size:
        msg_len, code, segment, security_id, sequence = HEADER.unpack_from(view, offset)
        body = offset + HEADER.size

        if code == FeedCode.BID or code == FeedCode.ASK:
            # Depth packets have a fixed size, so it is used instead of msg_len,
            # whose header-inclusive semantics the Dhan docs leave ambiguous.
            if end - offset < DEPTH_PACKET_SIZE:
                logger.warning(
                    "Truncated depth packet for security %d: %d of %d bytes",
                    security_id, end - offset, DEPTH_PACKET_SIZE,
                )
                break
            flat = LEVELS.unpack_from(view, body)
            messages.append(
                DepthUpdate(
                    kind="depth",
                    side="bid" if code == FeedCode.BID else "ask",
                    exchange_segment=segment,
                    security_id=security_id,
                    sequence=sequence,
                    recv_ns=recv_ns,
                    prices=list(flat[0::3]),
                    quantities=list(flat[1::3]),
                    orders=list(flat[2::3]),
                )
            )
            offset += DEPTH_PACKET_SIZE

        elif code == FeedCode.DISCONNECT:
            if end - offset < DISCONNECT_PACKET_SIZE:
                logger.warning("Truncated disconnect packet: %d bytes", end - offset)
                break
            (reason_code,) = DISCONNECT_REASON.unpack_from(view, body)
            messages.append(
                DisconnectNotice(
                    kind="disconnect",
                    exchange_segment=segment,
                    security_id=security_id,
                    reason_code=reason_code,
                    reason=DISCONNECT_REASONS.get(reason_code, "Unknown reason"),
                    recv_ns=recv_ns,
                )
            )
            offset += DISCONNECT_PACKET_SIZE

        else:
            # Skip unknown packet types by their declared length when it is sane.
            if HEADER.size <= msg_len <= end - offset:
                logger.debug("Skipping unknown packet code %d (%d bytes)", code, msg_len)
                offset += msg_len
                continue
            logger.warning(
                "Unknown packet code %d with unusable length %d; dropping %d bytes",
                code, msg_len, end - offset,
            )
            return messages

    if offset < end:
        logger.warning("Discarding %d trailing bytes", end - offset)
    return messages


def push_latest(queue: asyncio.Queue[T], item: T) -> bool:
    """Enqueue without blocking, evicting the oldest item if the queue is full.

    Stale book updates are worse than missing ones, so the producer never waits
    on a slow consumer. Returns False if an item was evicted.
    """
    try:
        queue.put_nowait(item)
        return True
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        queue.put_nowait(item)
        return False
