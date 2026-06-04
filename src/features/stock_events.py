from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StockEventSignalResult:
    score: float
    label: str
    bullish_hits: int
    bearish_hits: int
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "label": self.label,
            "bullish_hits": self.bullish_hits,
            "bearish_hits": self.bearish_hits,
            "note": self.note,
        }


BULLISH_PATTERNS = [
    "beat", "raises guidance", "guidance raised", "approval",
    "upgrade", "buyback", "contract", "record", "strong demand",
]
BEARISH_PATTERNS = [
    "cuts guidance", "guidance cut", "warning", "downgrade",
    "weak demand", "lawsuit", "probe", "delay", "miss", "recall",
]


def compute_stock_event_signal(
    headlines: pd.DataFrame | None,
    symbol: str,
    cfg: dict | None = None,
) -> StockEventSignalResult:
    cfg = cfg or {}
    bullish_threshold = float(cfg.get("bullish_threshold", 55))
    bearish_threshold = float(cfg.get("bearish_threshold", 45))
    strong_keyword_weight = float(cfg.get("strong_keyword_weight", 1.4))

    if headlines is None or headlines.empty or "title" not in headlines.columns:
        return StockEventSignalResult(50.0, "NEUTRAL", 0, 0, "No company event data")

    frame = headlines.copy()
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].fillna("").astype(str).str.upper()
        frame = frame[frame["symbol"] == symbol.upper()]
        if frame.empty:
            return StockEventSignalResult(50.0, "NEUTRAL", 0, 0, "No company event data")

    titles = frame["title"].astype(str).str.lower().fillna("")
    bullish_hits = 0
    bearish_hits = 0

    for title in titles:
        bullish_hits += sum(1 if token in title else 0 for token in BULLISH_PATTERNS)
        bearish_hits += sum(1 if token in title else 0 for token in BEARISH_PATTERNS)

    total_hits = bullish_hits + bearish_hits
    if total_hits == 0:
        return StockEventSignalResult(50.0, "NEUTRAL", 0, 0, "Events neutral")

    balance = (bullish_hits - bearish_hits * strong_keyword_weight) / total_hits
    score = max(0.0, min(100.0, 50.0 + balance * 25.0))
    if score >= bullish_threshold:
        label = "BULLISH"
    elif score <= bearish_threshold:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return StockEventSignalResult(
        score=round(score, 2),
        label=label,
        bullish_hits=bullish_hits,
        bearish_hits=bearish_hits,
        note=f"Company events: {bullish_hits} bullish / {bearish_hits} bearish hits",
    )
