"""Recorder -> trainer -> predictor on synthetic sessions with a planted signal."""

import io
import shutil
from datetime import date, datetime, time

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
from rich.console import Console
from sklearn.metrics import balanced_accuracy_score

from src.features.calculator import FEATURE_INDEX as F
from src.features.calculator import FEATURE_NAMES, MODEL_FEATURES
from src.features.orderbook import OrderBook
from src.inference.predictor import SignalPredictor
from src.pipeline.recorder import IST, ParquetRecorder
from src.pipeline.trainer import TrainConfig, add_labels, load_recordings, train
from tests.conftest import SESSIONS


def _ns(day: date, hh: int, mm: int, ss: int = 0) -> int:
    return int(datetime.combine(day, time(hh, mm, ss), tzinfo=IST).timestamp() * 1e9)


@pytest.mark.slow
def test_recorder_partitions_by_ist_date(recorded_dir):
    files = sorted(recorded_dir.rglob("*.parquet"))
    assert {f.parent.name for f in files} == {d.isoformat() for d in SESSIONS}
    assert not list(recorded_dir.rglob("*.tmp"))
    df = pl.read_parquet(files[0])
    assert df.columns[:3] == ["recv_ns", "exchange_segment", "security_id"]
    assert set(FEATURE_NAMES) <= set(df.columns)
    assert df["bid_px"].dtype == pl.Array(pl.Float64, 20)
    assert all(pl.read_parquet(f)["recv_ns"].is_sorted() for f in files)


def test_recorder_splits_a_buffer_spanning_midnight(tmp_path):
    rec = ParquetRecorder(tmp_path)
    book = OrderBook(1, 1333)
    features = np.zeros(len(FEATURE_NAMES))
    rec.append(_ns(date(2026, 10, 20), 23, 59, 59), book, features)
    rec.append(_ns(date(2026, 10, 21), 0, 0, 1), book, features)
    rec.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-10-20", "2026-10-21"]
    assert rec.rows_written == 2 and rec.files_written == 2


@pytest.mark.slow
def test_labels_match_brute_force(recorded_dir):
    cfg = TrainConfig(data_dir=recorded_dir, model_path=recorded_dir / "unused.txt")
    raw = load_recordings(recorded_dir)
    labeled = add_labels(raw, cfg)
    recv, mids = raw["recv_ns"].to_numpy(), raw["mid_price"].to_numpy()
    horizon = int(cfg.horizon_s * 1e9)
    for row in labeled.sample(300, seed=0).iter_rows(named=True):
        j = np.searchsorted(recv, row["recv_ns"] + horizon, side="left")
        assert recv[j] - (row["recv_ns"] + horizon) <= cfg.max_label_lag_s * 1e9
        ret = mids[j] / row["mid_price"] - 1
        expected = 1 if ret > cfg.threshold else -1 if ret < -cfg.threshold else 0
        assert row["label"] == expected
    assert labeled["session"].n_unique() == 2
    assert set(labeled["label"].unique().to_list()) == {-1, 0, 1}


@pytest.mark.slow
def test_predictor_matches_raw_booster(model_path, recorded_dir):
    predictor = SignalPredictor(model_path, threshold=0.70)
    x = pl.read_parquet(sorted(recorded_dir.rglob("*.parquet"))[-1]).select(FEATURE_NAMES).to_numpy()[:200]
    ours = np.array([[p.p_down, p.p_flat, p.p_up] for p in map(predictor.predict_proba, x)])
    reference = lgb.Booster(model_file=str(model_path)).predict(x[:, [F[n] for n in MODEL_FEATURES]])
    assert np.allclose(ours, reference)
    assert np.allclose(ours.sum(axis=1), 1.0)
    signals = {predictor.predict_proba(row).signal for row in x}
    assert signals <= {"LONG", "SHORT", "FLAT"}


@pytest.mark.slow
def test_model_learns_planted_signal_out_of_sample(recorded_dir, tmp_path):
    """Train on day 1 only (single-session 13:00 split path), score on unseen day 2."""
    day1 = tmp_path / "day1"
    shutil.copytree(recorded_dir / SESSIONS[0].isoformat(), day1 / SESSIONS[0].isoformat())
    model = tmp_path / "day1.txt"
    train(TrainConfig(data_dir=day1, model_path=model), Console(file=io.StringIO()))

    cfg = TrainConfig(data_dir=recorded_dir, model_path=model)
    day2 = add_labels(load_recordings(recorded_dir), cfg).filter(pl.col("session") == SESSIONS[1])
    proba = lgb.Booster(model_file=str(model)).predict(day2.select(MODEL_FEATURES).to_numpy())
    score = balanced_accuracy_score(day2["label"].to_numpy() + 1, proba.argmax(axis=1))
    assert score > 0.42, f"balanced accuracy {score:.3f} barely beats chance (0.333)"


def test_predictor_rejects_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        SignalPredictor(tmp_path / "nope.txt")


def test_predictor_rejects_model_with_other_features(tmp_path):
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((300, 3)), rng.integers(0, 3, 300)
    booster = lgb.train({"objective": "multiclass", "num_class": 3, "verbose": -1},
                        lgb.Dataset(x, y, feature_name=["a", "b", "c"]), num_boost_round=2)
    path = tmp_path / "other.txt"
    booster.save_model(str(path))
    with pytest.raises(ValueError, match="Retrain"):
        SignalPredictor(path)


def test_predictor_rejects_bad_threshold(tmp_path):
    with pytest.raises(ValueError, match="threshold"):
        SignalPredictor(tmp_path / "any.txt", threshold=0.4)
