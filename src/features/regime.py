"""
features/regime.py  —  L2 Market Regime
─────────────────────────────────────────────────────────────
判断市场环境：Green / Yellow / Red

Green  = 允许做新多头 swing
Yellow = 谨慎，减少新仓，已有仓位保留但不加仓
Red    = 停止新 long swing，优先平仓保利润

因子：
  - SPY > 200EMA           (P0, veto)
  - QQQ 21EMA > 50EMA      (P0, veto)
  - VIX 水平与 5 日变动    (P0)
  - US10Y 趋势             (P1)
  - DXY 趋势               (P1)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    color:           str    # Green / Yellow / Red
    gate_pass:       bool   # False = 环境 veto，不允许新多头
    score:           float  # 0-100，用于后续加权
    spy_above_200:   bool
    qqq_21_gt_50:    bool
    vix_level:       float
    vix_5d_chg:      float
    vix_state:       str    # LOW / NORMAL / ELEVATED / EXTREME
    us10y_5d_bps:    float
    dxy_5d_pct:      float
    checks:          dict = field(default_factory=dict)
    note:            str = ""

    def to_dict(self) -> dict:
        return {
            "color":         self.color,
            "gate_pass":     self.gate_pass,
            "score":         self.score,
            "spy_above_200": self.spy_above_200,
            "qqq_21_gt_50":  self.qqq_21_gt_50,
            "vix_level":     self.vix_level,
            "vix_5d_chg":    self.vix_5d_chg,
            "vix_state":     self.vix_state,
            "us10y_5d_bps":  self.us10y_5d_bps,
            "dxy_5d_pct":    round(self.dxy_5d_pct * 100, 2),
            "checks":        self.checks,
            "note":          self.note,
        }


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_regime(
    etf_data: dict[str, pd.DataFrame],
    macro_df:  pd.DataFrame | None = None,
    cfg: dict | None = None,
) -> RegimeResult:
    """
    etf_data: PriceClient.fetch_bulk() 返回的字典
    macro_df: FredClient.fetch_series() 返回的 DataFrame（可选）
    cfg:      thresholds.yaml 中 regime 块
    """
    cfg = cfg or {}
    vix_red   = cfg.get("vix_red_threshold", 28)
    vix_warn  = cfg.get("vix_5d_change_yellow", 0.25)
    t10y_warn = cfg.get("us10y_5d_warn_bps", 15)
    dxy_warn  = cfg.get("dxy_5d_warn_pct", 0.015)

    checks: dict[str, bool | str] = {}
    notes:  list[str] = []

    # ── SPY > 200EMA ──────────────────────────────────────────
    spy_above_200 = False
    if "SPY" in etf_data and not etf_data["SPY"].empty:
        spy = etf_data["SPY"]["close"]
        ema200 = _ema(spy, 200)
        spy_above_200 = bool(spy.iloc[-1] > ema200.iloc[-1])
    checks["spy_above_200ema"] = spy_above_200
    if not spy_above_200:
        notes.append("SPY below 200EMA → bearish macro")

    # ── QQQ 21EMA > 50EMA ─────────────────────────────────────
    qqq_21_gt_50 = False
    if "QQQ" in etf_data and not etf_data["QQQ"].empty:
        qqq = etf_data["QQQ"]["close"]
        e21 = _ema(qqq, 21)
        e50 = _ema(qqq, 50)
        qqq_21_gt_50 = bool(e21.iloc[-1] > e50.iloc[-1])
    checks["qqq_21_gt_50"] = qqq_21_gt_50
    if not qqq_21_gt_50:
        notes.append("QQQ 21EMA < 50EMA → growth trend broken")

    # ── VIX ───────────────────────────────────────────────────
    vix_level  = 20.0
    vix_5d_chg = 0.0
    try:
        vix_df = yf.Ticker("^VIX").history(period="30d")
        if not vix_df.empty:
            vix_level  = float(vix_df["Close"].iloc[-1])
            vix_5d_ago = float(vix_df["Close"].iloc[-6]) if len(vix_df) >= 6 else vix_level
            vix_5d_chg = (vix_level - vix_5d_ago) / vix_5d_ago if vix_5d_ago > 0 else 0
    except Exception as e:
        logger.warning("VIX fetch failed: %s", e)

    if vix_level >= vix_red:
        vix_state = "EXTREME"
    elif vix_level >= 22:
        vix_state = "ELEVATED"
    elif vix_level >= 16:
        vix_state = "NORMAL"
    else:
        vix_state = "LOW"

    vix_ok = vix_level < vix_red and vix_5d_chg < vix_warn
    checks["vix_state_ok"] = vix_ok
    if not vix_ok:
        notes.append(f"VIX {vix_state} ({vix_level:.1f}, +{vix_5d_chg*100:.1f}% 5d)")

    # ── US10Y 趋势（可选）────────────────────────────────────
    us10y_5d_bps = 0.0
    us10y_ok     = True
    if macro_df is not None and "us10y_5d_chg" in macro_df.columns:
        us10y_5d_bps = float(macro_df["us10y_5d_chg"].iloc[-1] * 100)  # pct → bps
        us10y_ok = abs(us10y_5d_bps) < t10y_warn
        checks["us10y_trend_ok"] = us10y_ok
        if not us10y_ok:
            notes.append(f"US10Y moved {us10y_5d_bps:+.1f}bps in 5d → rate pressure")

    # ── DXY 趋势（可选）─────────────────────────────────────
    dxy_5d_pct = 0.0
    dxy_ok     = True
    if macro_df is not None and "dxy_pct5d" in macro_df.columns:
        dxy_5d_pct = float(macro_df["dxy_pct5d"].iloc[-1])
        dxy_ok = dxy_5d_pct < dxy_warn
        checks["dxy_trend_ok"] = dxy_ok
        if not dxy_ok:
            notes.append(f"DXY +{dxy_5d_pct*100:.1f}% in 5d → risk-off")

    # ── 汇总 ──────────────────────────────────────────────────
    # Hard veto：两个 P0 因子任一不过 → Red
    veto = not spy_above_200 or not qqq_21_gt_50 or vix_state == "EXTREME"

    pass_count = sum([
        spy_above_200, qqq_21_gt_50, vix_ok, us10y_ok, dxy_ok
    ])
    total = 5

    if veto:
        color = "Red"
        gate  = False
        score = 20.0
    elif pass_count >= 4:
        color = "Green"
        gate  = True
        score = 60 + (pass_count / total) * 40
    elif pass_count >= 3:
        color = "Yellow"
        gate  = True   # 允许持有，不建议新开
        score = 35 + (pass_count / total) * 25
    else:
        color = "Red"
        gate  = False
        score = max(0, pass_count / total * 30)

    return RegimeResult(
        color=color,
        gate_pass=gate,
        score=round(score, 1),
        spy_above_200=spy_above_200,
        qqq_21_gt_50=qqq_21_gt_50,
        vix_level=round(vix_level, 2),
        vix_5d_chg=round(vix_5d_chg, 4),
        vix_state=vix_state,
        us10y_5d_bps=round(us10y_5d_bps, 2),
        dxy_5d_pct=round(dxy_5d_pct, 4),
        checks=checks,
        note=" | ".join(notes) if notes else f"Regime {color}: {pass_count}/{total} checks passed",
    )
