"""
features/breadth.py  —  L3 Breadth & Leadership
─────────────────────────────────────────────────────────────
市场内部健康度 + 板块领涨分析

数据来源：
  - 自定义股票宇宙内部计算（不依赖外部 breadth 数据库）
  - 板块 ETF 相对强弱

输出：
  BreadthResult(
    score          = 0-100,
    level          = HEALTHY / MIXED / WEAK,
    pct_above_200  = 0-1,
    pct_above_50   = 0-1,
    ema21_gt_50    = 0-1,
    adv_ratio      = 0-1,
    nh_nl_ratio    = float,
    leading_sectors = [...],
    lagging_sectors = [...],
  )
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BreadthResult:
    score:           float   # 0-100
    level:           str     # HEALTHY / MIXED / WEAK
    pct_above_200:   float   # 宇宙内在200EMA上方比例
    pct_above_50:    float
    ema21_gt_50:     float   # 21EMA>50EMA的比例
    adv_ratio:       float   # 今日上涨家数比例
    nh_nl_ratio:     float   # 20日新高/新低比
    leading_sectors: list[str] = field(default_factory=list)
    lagging_sectors: list[str] = field(default_factory=list)
    note:            str = ""

    def to_dict(self) -> dict:
        return {
            "score":           round(self.score, 1),
            "level":           self.level,
            "pct_above_200ema": round(self.pct_above_200 * 100, 1),
            "pct_above_50ema":  round(self.pct_above_50  * 100, 1),
            "ema21_gt_50_pct":  round(self.ema21_gt_50   * 100, 1),
            "adv_ratio":        round(self.adv_ratio      * 100, 1),
            "nh_nl_ratio":      round(self.nh_nl_ratio, 2),
            "leading_sectors":  self.leading_sectors,
            "lagging_sectors":  self.lagging_sectors,
            "note":             self.note,
        }


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def compute_breadth(
    universe_prices: dict[str, pd.DataFrame],
    etf_prices:      dict[str, pd.DataFrame],
    sector_etfs:     list[str] | None = None,
    cfg:             dict | None = None,
) -> BreadthResult:
    """
    universe_prices: 目标宇宙的价格字典
    etf_prices:      包含 SPY 及各板块 ETF 的价格字典
    sector_etfs:     要纳入 leadership 分析的 ETF 列表
    """
    cfg = cfg or {}
    green_th  = cfg.get("score_green",  60)
    yellow_th = cfg.get("score_yellow", 40)

    if sector_etfs is None:
        sector_etfs = ["XLK","XLV","XLE","XLF","XLI","XLY","SMH"]

    # ── 构建收盘价矩阵 ────────────────────────────────────────
    closes: dict[str, pd.Series] = {}
    for sym, df in universe_prices.items():
        col = "close" if "close" in df.columns else "Close"
        if col in df.columns and len(df) >= 200:
            closes[sym] = df[col]

    if not closes:
        return BreadthResult(
            score=0, level="WEAK",
            pct_above_200=0, pct_above_50=0,
            ema21_gt_50=0, adv_ratio=0, nh_nl_ratio=0,
            note="No universe price data available",
        )

    price_df = pd.DataFrame(closes).sort_index().ffill()

    # ── EMA 矩阵 ─────────────────────────────────────────────
    ema200 = price_df.apply(lambda c: _ema(c, 200), axis=0)
    ema50  = price_df.apply(lambda c: _ema(c, 50),  axis=0)
    ema21  = price_df.apply(lambda c: _ema(c, 21),  axis=0)

    last = price_df.iloc[-1]
    e200_last = ema200.iloc[-1]
    e50_last  = ema50.iloc[-1]
    e21_last  = ema21.iloc[-1]

    valid_mask = last.notna() & e200_last.notna()

    pct_above_200 = float((last[valid_mask] > e200_last[valid_mask]).mean())
    pct_above_50  = float((last[valid_mask] > e50_last[valid_mask]).mean())
    ema21_gt_50   = float((e21_last[valid_mask] > e50_last[valid_mask]).mean())

    # ── 当日涨跌比 ────────────────────────────────────────────
    daily_ret  = price_df.pct_change().iloc[-1]
    adv_ratio  = float((daily_ret[valid_mask] > 0).mean())

    # ── 20日新高/新低比 ──────────────────────────────────────
    high20 = price_df.rolling(20).max().iloc[-1]
    low20  = price_df.rolling(20).min().iloc[-1]
    at_high = float((last[valid_mask] >= high20[valid_mask] * 0.99).mean())
    at_low  = float((last[valid_mask] <= low20[valid_mask]  * 1.01).mean())
    nh_nl_ratio = at_high / (at_low + 1e-6)

    # ── 综合 Breadth Score ───────────────────────────────────
    score = (
        pct_above_200 * 30 +
        pct_above_50  * 25 +
        ema21_gt_50   * 25 +
        adv_ratio     * 20
    ) * 100
    score = float(np.clip(score, 0, 100))

    # ── 板块领涨/落后 ────────────────────────────────────────
    leading:  list[str] = []
    lagging:  list[str] = []

    spy_col = "close" if "close" in etf_prices.get("SPY", pd.DataFrame()).columns else "Close"
    spy_ret20 = None
    if "SPY" in etf_prices and not etf_prices["SPY"].empty:
        spy_s = etf_prices["SPY"][spy_col]
        if len(spy_s) >= 21:
            spy_ret20 = float(spy_s.iloc[-1] / spy_s.iloc[-21] - 1)

    for etf in sector_etfs:
        if etf not in etf_prices or etf_prices[etf].empty:
            continue
        col = "close" if "close" in etf_prices[etf].columns else "Close"
        s = etf_prices[etf][col]
        if len(s) < 21:
            continue
        ret20 = float(s.iloc[-1] / s.iloc[-21] - 1)
        if spy_ret20 is not None:
            rs = ret20 - spy_ret20
            if rs > 0.02:
                leading.append(etf)
            elif rs < -0.02:
                lagging.append(etf)

    # ── 等级 ─────────────────────────────────────────────────
    level = "HEALTHY" if score >= green_th else ("MIXED" if score >= yellow_th else "WEAK")

    notes = []
    if pct_above_200 < 0.4:
        notes.append(f"Only {pct_above_200*100:.0f}% above 200EMA")
    if nh_nl_ratio < 1.0:
        notes.append("More new lows than new highs")
    if leading:
        notes.append(f"Leaders: {','.join(leading)}")

    return BreadthResult(
        score=round(score, 1),
        level=level,
        pct_above_200=round(pct_above_200, 4),
        pct_above_50=round(pct_above_50, 4),
        ema21_gt_50=round(ema21_gt_50, 4),
        adv_ratio=round(adv_ratio, 4),
        nh_nl_ratio=round(nh_nl_ratio, 2),
        leading_sectors=leading,
        lagging_sectors=lagging,
        note=" | ".join(notes) if notes else f"Breadth {level}: score={score:.0f}",
    )
