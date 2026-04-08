"""
features/sentiment.py  —  L4 Sentiment
─────────────────────────────────────────────────────────────
轻量级情绪打分（-5 ~ +5），只做辅助加减分，不做 veto。

v1 方案：纯规则型关键词打分（不依赖 NLP 模型）
v1.1 预留接口：FinBERT 接入后替换 _rule_score()

来源：
  - RSS headlines（全局市场情绪）
  - Finnhub 公司新闻（个股情绪，可选）
  - VIX 趋势（恐慌代理）
  - Fear & Greed Index CNN（可选，偶发超时默认 50）
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    market_score:  float   # -5 ~ +5，全局市场情绪
    vix_score:     float   # -5 ~ +5，基于 VIX 趋势
    fg_score:      float   # 0-100，CNN Fear & Greed
    combined:      float   # 最终加减分 -5 ~ +5
    note:          str = ""

    def to_dict(self) -> dict:
        return {
            "market_sentiment_score": round(self.market_score, 2),
            "vix_sentiment_score":    round(self.vix_score, 2),
            "fear_greed_index":       round(self.fg_score, 1),
            "sentiment_adjustment":   round(self.combined, 2),
            "note":                   self.note,
        }


# ── 规则型关键词打分 ───────────────────────────────────────────
BULL_WORDS = [
    "surge", "rally", "beat", "record", "strong", "bullish",
    "breakout", "upgraded", "buy", "outperform", "growth",
    "positive", "rise", "gain", "recovery", "momentum",
    "robust", "resilient", "expansion",
]
BEAR_WORDS = [
    "crash", "plunge", "miss", "layoff", "recession", "bearish",
    "downgrade", "sell", "underperform", "weak", "concern",
    "decline", "fall", "risk", "warning", "cut", "loss",
    "contraction", "default", "bankruptcy", "tariff", "sanctions",
]


def _rule_score(headlines: pd.DataFrame) -> float:
    """规则型情绪打分，-5 ~ +5"""
    if headlines.empty or "title" not in headlines.columns:
        return 0.0
    titles = headlines["title"].str.lower().fillna("")
    bull = titles.apply(lambda t: sum(w in t for w in BULL_WORDS)).sum()
    bear = titles.apply(lambda t: sum(w in t for w in BEAR_WORDS)).sum()
    total = bull + bear
    if total == 0:
        return 0.0
    ratio = (bull - bear) / total
    return round(float(ratio) * 5, 2)


def _fetch_fear_greed() -> float:
    """CNN Fear & Greed Index，失败返回 50（中性）"""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        data = r.json()
        return float(data["fear_and_greed"]["score"])
    except Exception as e:
        logger.debug("Fear & Greed fetch failed: %s", e)
        return 50.0


def _vix_to_score(vix_level: float, vix_5d_chg: float) -> float:
    """
    将 VIX 状态转换为情绪分。
    低 VIX 稳定 → 看多加分；高 VIX 且快速上升 → 扣分
    """
    if vix_level > 30:
        base = -4.0
    elif vix_level > 24:
        base = -2.0
    elif vix_level > 18:
        base = 0.0
    elif vix_level > 14:
        base = 1.5
    else:
        base = 3.0

    # 5 日变化调整
    if vix_5d_chg > 0.30:
        base -= 2.0
    elif vix_5d_chg > 0.15:
        base -= 1.0
    elif vix_5d_chg < -0.15:
        base += 1.0

    return round(max(-5.0, min(5.0, base)), 2)


def compute_sentiment(
    headlines:    pd.DataFrame | None = None,
    vix_level:    float = 20.0,
    vix_5d_chg:   float = 0.0,
    fetch_fg:     bool  = True,
) -> SentimentResult:
    """
    计算综合情绪分。

    headlines: NewsClient.fetch_rss_headlines() 的输出
    vix_level/vix_5d_chg: 由 regime.py 传入，避免重复拉取
    """
    # 新闻情绪
    market_score = 0.0
    if headlines is not None:
        market_score = _rule_score(headlines)

    # VIX 情绪
    vix_score = _vix_to_score(vix_level, vix_5d_chg)

    # Fear & Greed
    fg_score = _fetch_fear_greed() if fetch_fg else 50.0

    # Fear & Greed 转换为 -5 ~ +5
    fg_adj = (fg_score - 50) / 10  # 50→0, 80→+3, 20→-3
    fg_adj = round(max(-3.0, min(3.0, fg_adj)), 2)

    # 加权合并（情绪分只占系统总分 5%，权重压低）
    combined = round(
        market_score * 0.30 +
        vix_score    * 0.50 +
        fg_adj       * 0.20,
        2
    )
    combined = max(-5.0, min(5.0, combined))

    notes = []
    if abs(market_score) >= 3:
        notes.append(f"Headlines strongly {'bullish' if market_score > 0 else 'bearish'}")
    if vix_score <= -3:
        notes.append("VIX spike — elevated fear")
    if fg_score <= 25:
        notes.append(f"Extreme Fear (F&G={fg_score:.0f})")
    elif fg_score >= 75:
        notes.append(f"Extreme Greed (F&G={fg_score:.0f})")

    return SentimentResult(
        market_score=market_score,
        vix_score=vix_score,
        fg_score=fg_score,
        combined=combined,
        note=" | ".join(notes) if notes else f"Sentiment neutral ({combined:+.1f})",
    )
