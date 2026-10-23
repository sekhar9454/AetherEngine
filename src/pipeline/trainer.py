"""Walk-forward LightGBM training on recorded Parquet data.

    python -m src.pipeline.trainer --data data --model model.txt

Labels: the mid-price ``horizon`` seconds ahead (first tick at or after t+h,
within the same session) versus now. Return > +threshold -> 1 (UP),
< -threshold -> -1 (DOWN), else 0 (FLAT).

Validation is strictly chronological, never shuffled:
  * one session:    train before --split-time (default 13:00 IST), test after.
  * several:        expanding window, each session tested on a model trained
                    only on the sessions before it.
Training rows whose label window reaches into the test period are purged.
The final model is fit on all data with the median best iteration from the folds.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from ..features.calculator import MODEL_FEATURES

logger = logging.getLogger(__name__)

IST: Final = ZoneInfo("Asia/Kolkata")
CLASSES: Final = (-1, 0, 1)  # label values; their position is the class index the model sees
CLASS_NAMES: Final = ("DOWN", "FLAT", "UP")
MIN_TRAIN_ROWS: Final = 1_000


@dataclass(frozen=True, slots=True)
class TrainConfig:
    data_dir: Path
    model_path: Path
    horizon_s: float = 3.0
    threshold: float = 0.0001  # 0.01%
    max_label_lag_s: float = 2.0  # drop rows whose next tick after t+h is later than this
    split_time: time = time(13, 0)
    security_id: int | None = None
    signal_threshold: float = 0.70
    validation_frac: float = 0.15
    max_estimators: int = 2_000
    early_stopping_rounds: int = 100


def load_recordings(data_dir: Path, security_id: int | None = None) -> pl.DataFrame:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {data_dir}")
    lf = pl.scan_parquet(files).select("recv_ns", "security_id", "mid_price", *MODEL_FEATURES)
    if security_id is not None:
        lf = lf.filter(pl.col("security_id") == security_id)
    df = lf.sort("recv_ns").collect()
    if df.is_empty():
        raise ValueError(f"No rows for security {security_id} in {data_dir}")
    return df.with_columns(
        session=pl.from_epoch("recv_ns", time_unit="ns")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Asia/Kolkata")
        .dt.date()
    )


def add_labels(df: pl.DataFrame, cfg: TrainConfig) -> pl.DataFrame:
    horizon_ns = int(cfg.horizon_s * 1e9)
    future = df.select(
        "security_id", "session",
        future_ns=pl.col("recv_ns"),
        future_mid=pl.col("mid_price"),
    )
    labeled = (
        df.with_columns(target_ns=pl.col("recv_ns") + horizon_ns)
        .join_asof(
            future,
            left_on="target_ns",
            right_on="future_ns",
            by=["security_id", "session"],
            strategy="forward",
            tolerance=int(cfg.max_label_lag_s * 1e9),
            check_sortedness=False,  # sorted by recv_ns, hence by target_ns, within every group
        )
        .drop_nulls("future_mid")
        .with_columns(ret=pl.col("future_mid") / pl.col("mid_price") - 1.0)
        .with_columns(
            label=pl.when(pl.col("ret") > cfg.threshold).then(1)
            .when(pl.col("ret") < -cfg.threshold).then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )
    )
    return labeled.drop("target_ns", "future_ns")


def purged_split(df: pl.DataFrame, boundary_ns: int, purge_ns: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rows before ``boundary_ns`` whose labels cannot see past it, and rows at/after it."""
    before = df.filter(pl.col("recv_ns") < boundary_ns - purge_ns)
    after = df.filter(pl.col("recv_ns") >= boundary_ns)
    return before, after


def make_folds(df: pl.DataFrame, cfg: TrainConfig) -> list[tuple[str, int]]:
    """(name, test_start_ns) pairs; each fold trains on everything before test_start."""
    sessions = df["session"].unique().sort().to_list()
    if len(sessions) == 1:
        start = datetime.combine(sessions[0], cfg.split_time, tzinfo=IST)
        return [(f"{sessions[0]} after {cfg.split_time:%H:%M}", int(start.timestamp() * 1e9))]
    starts = df.group_by("session").agg(pl.col("recv_ns").min()).sort("session")
    return [(str(s), ns) for s, ns in starts.iter_rows()][1:]


def _matrix(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Features and class indices (0 = DOWN, 1 = FLAT, 2 = UP).

    Labels are encoded here rather than by LGBMClassifier: LightGBM 4.7's
    eval_X/eval_y path skips the classifier's label encoding, so -1 labels
    would reach the validation set unencoded and corrupt early stopping.
    """
    return df.select(MODEL_FEATURES).to_numpy(), df["label"].to_numpy().astype(np.int64) + 1


def _new_model(n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbose=-1,
    )


def _fit_with_early_stopping(train: pl.DataFrame, cfg: TrainConfig, purge_ns: int) -> lgb.LGBMClassifier:
    # Early stopping on the chronologically last slice of the training window, never on the test set.
    cut = train["recv_ns"][int(len(train) * (1 - cfg.validation_frac))]
    inner, val = purged_split(train, cut, purge_ns)
    x_in, y_in = _matrix(inner)
    x_val, y_val = _matrix(val)
    model = _new_model(cfg.max_estimators)
    model.fit(
        x_in, y_in,
        eval_X=(x_val,),
        eval_y=(y_val,),
        feature_name=list(MODEL_FEATURES),
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )
    return model


@dataclass(frozen=True, slots=True)
class FoldResult:
    name: str
    train_rows: int
    test_rows: int
    best_iteration: int
    balanced_accuracy: float
    long_signals: int
    long_precision: float
    short_signals: int
    short_precision: float
    confusion: np.ndarray


def evaluate(name: str, model: lgb.LGBMClassifier, train_rows: int, test: pl.DataFrame,
             signal_threshold: float) -> FoldResult:
    x, y = _matrix(test)
    proba = model.booster_.predict(x)  # same path as live inference; columns DOWN, FLAT, UP
    pred = proba.argmax(axis=1)
    long_mask = proba[:, 2] > signal_threshold
    short_mask = proba[:, 0] > signal_threshold
    return FoldResult(
        name=name,
        train_rows=train_rows,
        test_rows=len(test),
        best_iteration=model.best_iteration_ or model.n_estimators,
        balanced_accuracy=balanced_accuracy_score(y, pred),
        long_signals=int(long_mask.sum()),
        long_precision=float((y[long_mask] == 2).mean()) if long_mask.any() else float("nan"),
        short_signals=int(short_mask.sum()),
        short_precision=float((y[short_mask] == 0).mean()) if short_mask.any() else float("nan"),
        confusion=confusion_matrix(y, pred, labels=range(len(CLASSES))),
    )


def _has_all_classes(df: pl.DataFrame) -> bool:
    return df["label"].n_unique() == len(CLASSES)


def train(cfg: TrainConfig, console: Console) -> lgb.Booster:
    df = add_labels(load_recordings(cfg.data_dir, cfg.security_id), cfg)
    if df["security_id"].n_unique() > 1:
        console.print(
            f"[yellow]Training one model on {df['security_id'].n_unique()} instruments; "
            "pass --security-id to train per instrument.[/]"
        )
    purge_ns = int((cfg.horizon_s + cfg.max_label_lag_s) * 1e9)
    _print_label_summary(df, console)

    results: list[FoldResult] = []
    for name, test_start in make_folds(df, cfg):
        train_df, test_df = purged_split(df, test_start, purge_ns)
        if len(train_df) < MIN_TRAIN_ROWS or test_df.is_empty() or not _has_all_classes(train_df):
            console.print(f"[yellow]Skipping fold {name}: insufficient train/test data or missing classes[/]")
            continue
        model = _fit_with_early_stopping(train_df, cfg, purge_ns)
        results.append(evaluate(name, model, len(train_df), test_df, cfg.signal_threshold))

    _print_folds(results, cfg.signal_threshold, console)

    if len(df) < MIN_TRAIN_ROWS or not _has_all_classes(df):
        raise ValueError(
            f"Need at least {MIN_TRAIN_ROWS} labelled rows covering UP, FLAT and DOWN; got {len(df)} rows, "
            f"classes {sorted(df['label'].unique().to_list())}. Record more data."
        )
    if results:
        n_estimators = max(1, int(statistics.median(r.best_iteration for r in results)))
        x, y = _matrix(df)
        final = _new_model(n_estimators)
        final.fit(x, y, feature_name=list(MODEL_FEATURES))
    else:
        console.print("[yellow]No usable validation folds; early-stopping on the last slice of all data.[/]")
        final = _fit_with_early_stopping(df, cfg, purge_ns)
        n_estimators = final.best_iteration_ or final.n_estimators

    if tuple(final.classes_) != tuple(range(len(CLASSES))):
        raise RuntimeError(f"Unexpected class order {final.classes_}")
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    final.booster_.save_model(str(cfg.model_path), num_iteration=n_estimators)
    console.print(
        f"[green]Saved final model ({n_estimators} rounds, {len(df):,} rows) to {cfg.model_path}[/]"
    )
    _print_importance(final.booster_, console)
    return final.booster_


def _print_label_summary(df: pl.DataFrame, console: Console) -> None:
    counts = dict(df.group_by("label").len().iter_rows())
    total = len(df)
    sessions = df["session"].n_unique()
    parts = ", ".join(
        f"{n}: {counts.get(c, 0):,} ({counts.get(c, 0) / total:.1%})" for c, n in zip(CLASSES, CLASS_NAMES)
    )
    console.print(f"{total:,} labelled rows over {sessions} session(s) - {parts}")


def _print_folds(results: list[FoldResult], threshold: float, console: Console) -> None:
    if not results:
        return
    table = Table(title=f"Walk-forward out-of-sample results (signal threshold {threshold:.2f})")
    for col in ("Test fold", "Train rows", "Test rows", "Rounds", "Bal. acc",
                "LONG n", "LONG prec", "SHORT n", "SHORT prec"):
        table.add_column(col, justify="right")
    pct = lambda v: "-" if np.isnan(v) else f"{v:.1%}"  # noqa: E731
    for r in results:
        table.add_row(
            r.name, f"{r.train_rows:,}", f"{r.test_rows:,}", str(r.best_iteration),
            f"{r.balanced_accuracy:.3f}", f"{r.long_signals:,}", pct(r.long_precision),
            f"{r.short_signals:,}", pct(r.short_precision),
        )
    console.print(table)
    total = sum(r.confusion for r in results)
    cm = Table(title="Pooled confusion matrix (rows = actual, cols = predicted)")
    cm.add_column("")
    for n in CLASS_NAMES:
        cm.add_column(n, justify="right")
    for n, row in zip(CLASS_NAMES, total):
        cm.add_row(n, *(f"{v:,}" for v in row))
    console.print(cm)
    console.print("Balanced accuracy of a random guess is 0.333.")


def _print_importance(booster: lgb.Booster, console: Console) -> None:
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1]
    total = gain.sum() or 1.0
    table = Table(title="Feature importance (gain)")
    table.add_column("Feature")
    table.add_column("Share", justify="right")
    for i in order:
        table.add_row(booster.feature_name()[i], f"{gain[i] / total:.1%}")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"), help="recorder output directory")
    parser.add_argument("--model", type=Path, default=Path("model.txt"))
    parser.add_argument("--security-id", type=int, default=None, help="train on one instrument only")
    parser.add_argument("--horizon", type=float, default=3.0, help="label look-ahead in seconds")
    parser.add_argument("--threshold-pct", type=float, default=0.01, help="UP/DOWN move threshold in percent")
    parser.add_argument("--split-time", type=time.fromisoformat, default=time(13, 0),
                        help="IST train/test boundary when only one session is recorded")
    parser.add_argument("--signal-threshold", type=float, default=0.70)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = TrainConfig(
        data_dir=args.data,
        model_path=args.model,
        horizon_s=args.horizon,
        threshold=args.threshold_pct / 100,
        split_time=args.split_time,
        security_id=args.security_id,
        signal_threshold=args.signal_threshold,
    )
    train(cfg, Console())


if __name__ == "__main__":
    main()
