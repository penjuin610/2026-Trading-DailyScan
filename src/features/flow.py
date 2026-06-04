from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FlowSignalResult:
    score: float
    label: str
    net_inflow: float
    anomaly_count: int
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "label": self.label,
            "net_inflow": round(self.net_inflow, 2),
            "anomaly_count": self.anomaly_count,
            "note": self.note,
        }


def compute_flow_signal(
    flows: pd.DataFrame | None,
    symbol: str,
    cfg: dict | None = None,
) -> FlowSignalResult:
    cfg = cfg or {}
    bullish_threshold = float(cfg.get("bullish_threshold", 55))
    bearish_threshold = float(cfg.get("bearish_threshold", 45))
    scale = float(cfg.get("net_inflow_scale", 2_000_000))
    anomaly_bonus = float(cfg.get("anomaly_bonus", 10))

    if flows is None or flows.empty:
        return FlowSignalResult(50.0, "NEUTRAL", 0.0, 0, "No flow snapshots loaded")

    frame = flows.copy()
    frame["symbol"] = frame["symbol"].fillna("").astype(str).str.upper()
    bucket = frame[frame["symbol"] == symbol.upper()]
    if bucket.empty:
        return FlowSignalResult(50.0, "NEUTRAL", 0.0, 0, "No symbol flow data")

    net_inflow = float(bucket["net_inflow"].fillna(0).sum())
    anomaly_count = int(bucket["anomaly"].fillna(False).astype(bool).sum())
    flow_balance = max(-25.0, min(25.0, (net_inflow / scale) * 10.0))
    score = 50.0 + flow_balance
    if net_inflow > 0:
        score += min(15.0, anomaly_count * anomaly_bonus)
    elif net_inflow < 0:
        score -= min(15.0, anomaly_count * anomaly_bonus)
    score = max(0.0, min(100.0, score))

    if score >= bullish_threshold:
        label = "BULLISH"
    elif score <= bearish_threshold:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return FlowSignalResult(
        score=round(score, 2),
        label=label,
        net_inflow=net_inflow,
        anomaly_count=anomaly_count,
        note=f"Net inflow {net_inflow:,.0f} across {len(bucket)} snapshots",
    )
