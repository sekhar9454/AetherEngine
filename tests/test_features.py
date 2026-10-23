"""Order book state and feature math on a hand-built book with known answers."""

import pytest

from src.features.calculator import FEATURE_INDEX as F
from src.features.calculator import FEATURE_NAMES, MODEL_FEATURES, compute_features
from src.features.orderbook import BookRegistry, OrderBook

BID_PX = [99.99, 99.98, 99.95, 99.90, 99.80] + [99.0] * 14 + [0.0]
ASK_PX = [100.01, 100.02, 100.05, 100.10, 100.20] + [101.0] * 14 + [0.0]


def _update(side, prices, quantities, recv_ns=1):
    return {"kind": "depth", "side": side, "exchange_segment": 1, "security_id": 1333, "sequence": 0,
            "recv_ns": recv_ns, "prices": prices, "quantities": quantities, "orders": [1] * 20}


@pytest.fixture
def book():
    b = OrderBook(1, 1333)
    b.apply(_update("bid", BID_PX, [300, 100, 100, 100, 100] + [10] * 14 + [0]))
    b.apply(_update("ask", ASK_PX, [100, 100, 100, 100, 100] + [10] * 14 + [0]))
    return b


def test_one_sided_book_has_no_features():
    b = OrderBook(1, 1333)
    assert compute_features(b) is None
    b.apply(_update("bid", BID_PX, [100] * 20))
    assert not b.ready and compute_features(b) is None


def test_spread_mid_and_micro_price(book):
    x = compute_features(book)
    assert x[F["spread"]] == pytest.approx(0.02)
    assert x[F["mid_price"]] == pytest.approx(100.0)
    assert x[F["spread_bps"]] == pytest.approx(2.0)
    assert x[F["micro_price"]] == pytest.approx((300 * 100.01 + 100 * 99.99) / 400)


def test_order_book_imbalance(book):
    x = compute_features(book)
    assert x[F["obi_l1"]] == pytest.approx(0.5)
    assert x[F["obi_5"]] == pytest.approx((700 - 500) / 1200)


def test_spectrum_bands(book):
    # Bid distances from mid 100: .01%, .02%, .05% -> band 1; .10% -> band 2; .20% -> band 3; 1% -> outside.
    x = compute_features(book)
    assert [x[F[f"bid_band_{i}"]] for i in (1, 2, 3)] == [500, 100, 100]
    assert [x[F[f"ask_band_{i}"]] for i in (1, 2, 3)] == [300, 100, 100]
    assert x[F["band_1_imbalance"]] == pytest.approx(0.25)


def test_locked_book_is_skipped(book):
    book.apply(_update("ask", [99.99] + ASK_PX[1:], [100] * 20))
    assert compute_features(book) is None


def test_model_features_exclude_price_levels():
    assert "mid_price" not in MODEL_FEATURES and "micro_price" not in MODEL_FEATURES
    assert set(MODEL_FEATURES) < set(FEATURE_NAMES)


def test_registry_keys_by_segment_and_security():
    reg = BookRegistry()
    a = reg.apply(_update("bid", BID_PX, [1] * 20))
    other = dict(_update("bid", BID_PX, [1] * 20), exchange_segment=4)
    b = reg.apply(other)
    assert a is not b and len(reg) == 2
    assert reg.get((1, 1333)) is a
