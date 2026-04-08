"""
adapters/fred_client.py
─────────────────────────────────────────────────────────────
FRED 宏观时间序列适配器
免费 API key：https://fred.stlouisfed.org/docs/api/api_key.html

pip install fredapi
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 默认序列映射（全部是 FRED 公开免费序列）
DEFAULT_SERIES: dict[str, str] = {
    "us10y":  "DGS10",          # 10年期国债收益率（日频）
    "us2y":   "DGS2",           # 2年期
    "us10y2y_spread": None,     # 计算得出，不直接拉取
    "dxy":    "DTWEXBGS",       # 贸易加权美元（FRED官方，替代DXY）
    "oil_wti":"DCOILWTICO",     # WTI 原油（日频）
    "gold":   "GOLDAMGBD228NLBM", # 伦敦金定盘价
    "vix":    None,             # 从 yfinance 拉 ^VIX，FRED版本有延迟
    "fedfunds":"FEDFUNDS",      # 联邦基金利率（月频，用于背景判断）
    "cpi_yoy": "CPIAUCSL",      # CPI（月频，同比需自行计算）
}


class FredClient:
    """
    FRED 宏观数据适配器。
    FRED_API_KEY 优先从环境变量读取，其次从参数传入。
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "需要 FRED API key。\n"
                "1. 注册：https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "2. 设置环境变量：export FRED_API_KEY=your_key\n"
                "   或传入参数：FredClient(api_key='your_key')"
            )
        try:
            from fredapi import Fred
            self._fred = Fred(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install fredapi")

    def fetch_series(
        self,
        series_ids: dict[str, str] | None = None,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        批量拉取 FRED 序列，返回对齐后的 DataFrame。
        自动向前填充（FRED 有些序列是月频，需要前向填充到日频）。
        """
        if series_ids is None:
            series_ids = {k: v for k, v in DEFAULT_SERIES.items() if v is not None}

        frames: dict[str, pd.Series] = {}
        for name, sid in series_ids.items():
            try:
                s = self._fred.get_series(sid)
                frames[name] = s
                logger.debug("FRED OK: %s (%s)", name, sid)
            except Exception as e:
                logger.warning("FRED failed: %s (%s): %s", name, sid, e)

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = df.sort_index().tail(lookback_days * 2)  # 多拉一些用于计算斜率
        df = df.ffill()  # 月频序列前向填充
        df = df.dropna(how="all")

        # 派生列
        if "us10y" in df.columns and "us2y" in df.columns:
            df["us10y2y_spread"] = df["us10y"] - df["us2y"]

        # 变化率列（5日、20日）
        for col in ["us10y", "dxy", "oil_wti", "gold"]:
            if col in df.columns:
                df[f"{col}_5d_chg"] = df[col].diff(5)
                df[f"{col}_20d_chg"] = df[col].diff(20)
                df[f"{col}_pct5d"] = df[col].pct_change(5)

        return df.tail(lookback_days)

    def latest(self) -> dict[str, float]:
        """返回各序列最新值的简洁字典"""
        df = self.fetch_series(lookback_days=30)
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {k: round(float(v), 4) for k, v in row.items() if pd.notna(v)}
