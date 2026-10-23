"""Experiment parameters, validated once at the portal boundary."""

from __future__ import annotations

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..feed.binary_parser import ExchangeSegment


class Charges(BaseModel):
    """NSE equity intraday costs. Defaults approximate Dhan's published rates; verify before relying on them."""

    enabled: bool = True
    brokerage_pct: float = Field(0.03, ge=0, description="% of order value, per executed order")
    brokerage_cap: float = Field(20.0, ge=0, description="₹ maximum brokerage per order")
    stt_sell_pct: float = Field(0.025, ge=0, description="Securities transaction tax, sell side, %")
    exchange_pct: float = Field(0.00297, ge=0, description="NSE transaction charge, %")
    sebi_per_crore: float = Field(10.0, ge=0, description="SEBI turnover fee, ₹ per crore")
    stamp_buy_pct: float = Field(0.003, ge=0, description="Stamp duty, buy side, %")
    gst_pct: float = Field(18.0, ge=0, description="GST on brokerage and exchange/SEBI fees, %")


class ExperimentConfig(BaseModel):
    name: str = Field("", max_length=80)
    mode: Literal["replay", "live"]
    exchange_segment: str = "NSE_EQ"
    security_id: int = Field(gt=0)
    model: str = Field(min_length=1, description="model file name in the models directory")
    days: list[date] = Field(default_factory=list, description="replay only; empty means every recorded day")

    starting_capital: float = Field(100_000, gt=0)
    size_mode: Literal["shares", "value"] = "value"
    trade_size: float = Field(50_000, gt=0, description="shares, or ₹ per trade")
    leverage: float = Field(1.0, ge=1, le=5, description="max position value as a multiple of equity")
    allow_short: bool = True

    signal_threshold: float = Field(0.70, ge=0.5, lt=1)
    hold_seconds: float = Field(3.0, ge=0, description="time exit; 0 disables it")
    exit_on_opposite: bool = True
    cooldown_seconds: float = Field(0, ge=0, description="wait after an exit before a new entry")

    daily_loss_limit: float = Field(2_000, ge=0, description="₹; stop trading for the day past this loss, 0 = off")
    start_time: time = time(9, 15)
    square_off_time: time = time(15, 15)
    latency_ms: float = Field(0, ge=0, le=5_000, description="delay between a decision and its fill")

    charges: Charges = Field(default_factory=Charges)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if self.exchange_segment not in ExchangeSegment.__members__:
            raise ValueError(f"Unknown exchange segment {self.exchange_segment!r}")
        if self.start_time >= self.square_off_time:
            raise ValueError("start_time must be before square_off_time")
        if self.size_mode == "shares" and self.trade_size != int(self.trade_size):
            raise ValueError("trade_size must be a whole number of shares")
        if "/" in self.model or "\\" in self.model or self.model.startswith("."):
            raise ValueError("model must be a plain file name")
        return self
