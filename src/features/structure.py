"""
features/structure.py  —  L5 Structure
─────────────────────────────────────────────────────────────
个股趋势结构筛选 + Relative Strength 计算

筛选条件（全部可在 thresholds.yaml 调整）：
  1. Price > 200EMA
  2. 21EMA > 50EMA
  3. 50EMA 斜率向上（5日均线斜率为正）
  4. RS_20d vs SPY > 0（领涨）
  5. RS_20d vs Sector > -2%

输出：
  per-symbol StructureResult + 候选列表（按 RS 排序）
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StructureResult:
    symbol:          str
    close:           float
    ema21:           float
    ema50:           float
    ema200:          float
    above_200:       bool
    ema21_gt_50:     bool
    ema50_slope_up:  bool
    rs_vs_spy:       float   # % outperformance vs SPY (20d)
    rs_vs_sector:    float   # % outperformance vs sector ETF (20d)
    atr14:           float   # 用于止损计算
    structure_ok:    bool    # 全部条件通过
    score:           float   # 0-100，用于候选排序
    note:            str = ""

    def to_dict(self) -> dict:
        return {
            "symbol":         self.symbol,
            "close":          round(self.close, 2),
            "ema21":          round(self.ema21, 2),
            "ema50":          round(self.ema50, 2),
            "ema200":         round(self.ema200, 2),
            "above_200ema":   self.above_200,
            "ema21_gt_50":    self.ema21_gt_50,
            "ema50_slope_up": self.ema50_slope_up,
            "rs_vs_spy_pct":  round(self.rs_vs_spy * 100, 2),
            "rs_vs_sector_pct": round(self.rs_vs_sector * 100, 2),
            "atr14":          round(self.atr14, 2),
            "structure_ok":   self.structure_ok,
            "score":          round(self.score, 1),
            "note":           self.note,
        }


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """计算最新 ATR14"""
    if len(df) < period + 1:
        return float(df["close"].std()) if "close" in df.columns else 0.0
    hi = df["high"] if "high" in df.columns else df["close"]
    lo = df["low"]  if "low"  in df.columns else df["close"]
    cl = df["close"]
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift(1)).abs(),
        (lo - cl.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def screen_structure(
    universe_prices:  dict[str, pd.DataFrame],
    etf_prices:       dict[str, pd.DataFrame],
    stock_sector_map: dict[str, str] | None = None,
    cfg:              dict | None = None,
) -> list[StructureResult]:
    """
    对整个宇宙做结构筛选，返回所有标的的 StructureResult 列表。
    candidates = [r for r in results if r.structure_ok]
    """
    cfg = cfg or {}
    ema50_slope_days = cfg.get("ema50_slope_days", 5)
    rs_spy_min       = cfg.get("rs_vs_spy_min", 0.0)
    rs_sector_min    = cfg.get("rs_vs_sector_min", -0.02)

    if stock_sector_map is None:
        stock_sector_map = {}

    # SPY 基准 20 日收益
    spy_ret20 = 0.0
    if "SPY" in etf_prices and not etf_prices["SPY"].empty:
        spy_cl = etf_prices["SPY"]["close"]
        if len(spy_cl) >= 21:
            spy_ret20 = float(spy_cl.iloc[-1] / spy_cl.iloc[-21] - 1)

    results: list[StructureResult] = []

    for sym, df in universe_prices.items():
        col = "close" if "close" in df.columns else "Close"
        if col not in df.columns or len(df) < 210:
            continue

        # 标准化列名
        if col == "Close":
            df = df.rename(columns={"Close":"close","High":"high",
                                     "Low":"low","Volume":"volume"})

        cl = df["close"]

        # EMA
        e21  = _ema(cl, 21)
        e50  = _ema(cl, 50)
        e200 = _ema(cl, 200)

        close_val  = float(cl.iloc[-1])
        ema21_val  = float(e21.iloc[-1])
        ema50_val  = float(e50.iloc[-1])
        ema200_val = float(e200.iloc[-1])

        above_200    = close_val > ema200_val
        ema21_gt_50  = ema21_val > ema50_val
        ema50_slope  = ema50_val > float(e50.iloc[-ema50_slope_days - 1])

        # RS vs SPY
        ret20 = float(cl.iloc[-1] / cl.iloc[-21] - 1) if len(cl) >= 21 else 0.0
        rs_vs_spy = ret20 - spy_ret20

        # RS vs Sector
        sector_etf = stock_sector_map.get(sym, "SPY")
        sector_ret20 = spy_ret20
        if sector_etf in etf_prices and not etf_prices[sector_etf].empty:
            sec_cl = etf_prices[sector_etf]["close"]
            if len(sec_cl) >= 21:
                sector_ret20 = float(sec_cl.iloc[-1] / sec_cl.iloc[-21] - 1)
        rs_vs_sector = ret20 - sector_ret20

        # ATR
        atr = _atr(df)

        # 全部条件
        structure_ok = (
            above_200 and
            ema21_gt_50 and
            ema50_slope and
            rs_vs_spy    >= rs_spy_min and
            rs_vs_sector >= rs_sector_min
        )

        # 评分（0-100，用于排序）
        score = 0.0
        score += 25 if above_200    else 0
        score += 20 if ema21_gt_50  else 0
        score += 15 if ema50_slope  else 0
        score += min(25, max(0, rs_vs_spy    * 500 + 12.5))  # RS vs SPY
        score += min(15, max(0, rs_vs_sector * 300 +  7.5))  # RS vs Sector

        notes = []
        if not above_200:   notes.append("below 200EMA")
        if not ema21_gt_50: notes.append("21EMA < 50EMA")
        if not ema50_slope: notes.append("50EMA declining")

        results.append(StructureResult(
            symbol=sym,
            close=close_val,
            ema21=ema21_val,
            ema50=ema50_val,
            ema200=ema200_val,
            above_200=above_200,
            ema21_gt_50=ema21_gt_50,
            ema50_slope_up=ema50_slope,
            rs_vs_spy=rs_vs_spy,
            rs_vs_sector=rs_vs_sector,
            atr14=atr,
            structure_ok=structure_ok,
            score=round(score, 1),
            note=", ".join(notes) if notes else "OK",
        ))

    # 按 RS vs SPY 降序
    results.sort(key=lambda r: r.rs_vs_spy, reverse=True)
    return results
