"""
features/triggers.py  —  L6 Trigger
─────────────────────────────────────────────────────────────
量价触发确认：健康回踩 + 再启动买点

买点逻辑（顺序执行）：
  1. 价格回踩至 21/50EMA 附近（3%/5% 以内）
  2. 回踩过程成交量萎缩（< 突破日量的 70%）
  3. 出现止跌 K 线（收盘价 > 开盘价，且收盘 > 前日低点）
  4. 再启动日量能回升（> 20日均量的 1.2 倍）

卖点逻辑：
  - 跌破回踩低点
  - 放量跌破 21/50EMA
  - 时间止损（入场后 5-7 日无启动）
  - 分批止盈（1R/2R）
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TriggerResult:
    symbol:            str
    signal:            str    # BUY_CANDIDATE / WATCH / PASS / EXIT
    pullback_near_21:  bool
    pullback_near_50:  bool
    volume_contracted: bool   # 回踩缩量
    restart_volume:    bool   # 再启动放量
    stop_below:        float  # 建议止损价
    entry_zone_low:    float  # 入场区低
    entry_zone_high:   float  # 入场区高
    score:             float  # 触发质量 0-100
    note:              str = ""

    def to_dict(self) -> dict:
        return {
            "symbol":           self.symbol,
            "signal":           self.signal,
            "pullback_near_21": self.pullback_near_21,
            "pullback_near_50": self.pullback_near_50,
            "volume_contracted":self.volume_contracted,
            "restart_volume":   self.restart_volume,
            "stop_below":       round(self.stop_below, 2),
            "entry_zone":       f"{self.entry_zone_low:.2f} - {self.entry_zone_high:.2f}",
            "score":            round(self.score, 1),
            "note":             self.note,
        }


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def check_trigger(
    symbol:    str,
    df:        pd.DataFrame,
    atr14:     float,
    cfg:       dict | None = None,
) -> TriggerResult:
    """
    对单只股票检查触发条件。
    df 需包含 close / high / low / volume (小写)
    atr14 由 structure.py 传入
    """
    cfg = cfg or {}
    pb_vol_max   = cfg.get("pullback_volume_ratio_max", 0.70)
    restart_min  = cfg.get("restart_volume_ratio_min",  1.20)
    near_21_pct  = cfg.get("pullback_near_ema21_pct",   0.03)
    near_50_pct  = cfg.get("pullback_near_ema50_pct",   0.05)

    col_map = {c.lower(): c for c in df.columns}
    cl  = df[col_map.get("close", "close")]
    vol = df[col_map.get("volume","volume")] if "volume" in col_map else pd.Series(dtype=float)

    e21  = _ema(cl, 21)
    e50  = _ema(cl, 50)
    vol20 = vol.rolling(20).mean() if not vol.empty else pd.Series(dtype=float)

    close_now = float(cl.iloc[-1])
    e21_now   = float(e21.iloc[-1])
    e50_now   = float(e50.iloc[-1])

    # ── 1. 是否接近 21/50 EMA ─────────────────────────────────
    near_21 = abs(close_now - e21_now) / e21_now < near_21_pct
    near_50 = abs(close_now - e50_now) / e50_now < near_50_pct

    # ── 2. 回踩缩量（过去 3 日量能均值 vs 20 日均量）──────────
    vol_contracted = False
    if not vol.empty and not vol20.empty and len(vol20.dropna()) > 0:
        recent_vol_avg = float(vol.iloc[-3:].mean())
        ma20_vol       = float(vol20.iloc[-1])
        if ma20_vol > 0:
            vol_ratio     = recent_vol_avg / ma20_vol
            vol_contracted = vol_ratio < pb_vol_max

    # ── 3. 今日是否启动放量 ────────────────────────────────────
    restart_vol = False
    if not vol.empty and not vol20.empty and len(vol20.dropna()) > 0:
        today_vol  = float(vol.iloc[-1])
        ma20_vol   = float(vol20.iloc[-1])
        if ma20_vol > 0:
            restart_vol = today_vol / ma20_vol >= restart_min

    # ── 4. 止跌 K 线（收盘 > 开盘，收盘 > 前日低）────────────
    reversal_candle = False
    if "open" in col_map and "low" in col_map:
        op  = df[col_map["open"]]
        lo  = df[col_map["low"]]
        if len(op) >= 2:
            reversal_candle = (
                float(cl.iloc[-1]) > float(op.iloc[-1]) and
                float(cl.iloc[-1]) > float(lo.iloc[-2])
            )

    # ── 止损价 ────────────────────────────────────────────────
    # 回踩低点 or ATR止损
    recent_low = float(cl.iloc[-5:].min()) if len(cl) >= 5 else close_now * 0.95
    stop_below = min(recent_low - atr14 * 0.5, close_now * 0.95)
    stop_below = max(stop_below, close_now * 0.85)  # 最多亏 15%

    # ── 入场区 ────────────────────────────────────────────────
    entry_low  = min(e21_now, e50_now) * 0.99
    entry_high = max(e21_now, e50_now) * 1.02

    # ── 信号判断 ─────────────────────────────────────────────
    notes = []

    if (near_21 or near_50) and vol_contracted and restart_vol:
        signal = "BUY_CANDIDATE"
        notes.append("Pullback + volume restart confirmed")
    elif (near_21 or near_50) and vol_contracted:
        signal = "WATCH"
        notes.append("Pullback OK, waiting for volume restart")
    elif close_now < e21_now * 0.97 and not vol_contracted:
        signal = "EXIT"
        notes.append("Price below 21EMA with volume expansion")
    else:
        signal = "PASS"
        notes.append("No clean setup")

    # ── 触发质量评分 ─────────────────────────────────────────
    score = 0.0
    score += 30 if (near_21 or near_50)  else 0
    score += 25 if vol_contracted        else 0
    score += 30 if restart_vol           else 0
    score += 15 if reversal_candle       else 0

    return TriggerResult(
        symbol=symbol,
        signal=signal,
        pullback_near_21=near_21,
        pullback_near_50=near_50,
        volume_contracted=vol_contracted,
        restart_volume=restart_vol,
        stop_below=stop_below,
        entry_zone_low=entry_low,
        entry_zone_high=entry_high,
        score=round(score, 1),
        note=" | ".join(notes),
    )


def screen_triggers(
    candidates:      list,   # list[StructureResult] from structure.py
    universe_prices: dict[str, pd.DataFrame],
    cfg:             dict | None = None,
) -> list[TriggerResult]:
    """对结构通过的候选做触发扫描"""
    results = []
    for cand in candidates:
        sym = cand.symbol
        if sym not in universe_prices:
            continue
        df = universe_prices[sym]
        # 标准化列名
        df = df.rename(columns={c: c.lower() for c in df.columns})
        tr = check_trigger(sym, df, atr14=cand.atr14, cfg=cfg)
        results.append(tr)

    results.sort(key=lambda r: r.score, reverse=True)
    return results
