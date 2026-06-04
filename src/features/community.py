from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CommunitySentimentResult:
    market_score: float
    market_label: str
    symbol_score: float
    symbol_label: str
    overall_score: float
    bullish_mentions: float
    bearish_mentions: float
    neutral_mentions: float
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "market_score": round(self.market_score, 2),
            "market_label": self.market_label,
            "symbol_score": round(self.symbol_score, 2),
            "symbol_label": self.symbol_label,
            "overall_score": round(self.overall_score, 2),
            "bullish_mentions": round(self.bullish_mentions, 2),
            "bearish_mentions": round(self.bearish_mentions, 2),
            "neutral_mentions": round(self.neutral_mentions, 2),
            "note": self.note,
        }


def _score_bucket(bucket: pd.DataFrame, bullish_threshold: float, bearish_threshold: float) -> tuple[float, str, float, float, float]:
    if bucket.empty:
        return 50.0, "NEUTRAL", 0.0, 0.0, 0.0

    mentions = bucket.get("mentions", pd.Series([1] * len(bucket))).fillna(1).astype(float)
    sentiments = bucket["sentiment"].str.lower().fillna("neutral")
    bullish = float(mentions[sentiments == "bullish"].sum())
    bearish = float(mentions[sentiments == "bearish"].sum())
    neutral = float(mentions[sentiments == "neutral"].sum())
    total = bullish + bearish + neutral
    if total <= 0:
        return 50.0, "NEUTRAL", 0.0, 0.0, 0.0

    balance = (bullish - bearish) / total
    score = max(0.0, min(100.0, 50.0 + balance * 50.0))
    if score >= bullish_threshold:
        label = "BULLISH"
    elif score <= bearish_threshold:
        label = "BEARISH"
    else:
        label = "NEUTRAL"
    return score, label, bullish, bearish, neutral


def compute_community_sentiment(
    posts: pd.DataFrame | None,
    symbol: str | None = None,
    cfg: dict | None = None,
) -> CommunitySentimentResult:
    cfg = cfg or {}
    market_symbol = str(cfg.get("market_symbol", "MARKET")).upper()
    bullish_threshold = float(cfg.get("bullish_threshold", 55))
    bearish_threshold = float(cfg.get("bearish_threshold", 45))
    symbol_weight = float(cfg.get("symbol_weight", 0.65))
    market_weight = float(cfg.get("market_weight", 0.35))

    if posts is None or posts.empty:
        return CommunitySentimentResult(
            market_score=50.0,
            market_label="NEUTRAL",
            symbol_score=50.0,
            symbol_label="NEUTRAL",
            overall_score=50.0,
            bullish_mentions=0.0,
            bearish_mentions=0.0,
            neutral_mentions=0.0,
            note="No community snapshots loaded",
        )

    frame = posts.copy()
    frame["symbol"] = frame["symbol"].fillna(market_symbol).astype(str).str.upper()

    market_bucket = frame[frame["symbol"] == market_symbol]
    market_score, market_label, market_bull, market_bear, market_neutral = _score_bucket(
        market_bucket,
        bullish_threshold,
        bearish_threshold,
    )

    symbol_bucket = frame[frame["symbol"] == symbol.upper()] if symbol else market_bucket
    symbol_score, symbol_label, symbol_bull, symbol_bear, symbol_neutral = _score_bucket(
        symbol_bucket,
        bullish_threshold,
        bearish_threshold,
    )

    overall_score = (
        market_score * market_weight + symbol_score * symbol_weight
        if symbol
        else market_score
    )
    return CommunitySentimentResult(
        market_score=round(market_score, 2),
        market_label=market_label,
        symbol_score=round(symbol_score, 2),
        symbol_label=symbol_label,
        overall_score=round(overall_score, 2),
        bullish_mentions=market_bull + symbol_bull,
        bearish_mentions=market_bear + symbol_bear,
        neutral_mentions=market_neutral + symbol_neutral,
        note=f"Community: market {market_label}, symbol {symbol_label}",
    )
