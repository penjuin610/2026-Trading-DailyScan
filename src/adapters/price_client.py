"""
adapters/price_client.py
─────────────────────────────────────────────────────────────
OHLCV + ETF 数据适配器（基于 yfinance，无 key，无流量限制）
替代原方案中的 finnhub_client.py

使用方式：
    from src.adapters.price_client import PriceClient
    client = PriceClient()
    df = client.fetch_single("NVDA", period="1y")
    bulk = client.fetch_bulk(["SPY","QQQ","NVDA"], period="2y")
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


class PriceClient:
    """
    yfinance 封装层。
    - 自动落地 CSV 缓存（减少重复拉取）
    - 批量下载时自动处理失败标的
    - 对外接口统一返回 OHLCV DataFrame
    """

    def __init__(self, cache: bool = True, cache_dir: Path = RAW_DIR):
        self.cache = cache
        self.cache_dir = cache_dir

    def _cache_path(self, symbol: str, period: str, interval: str) -> Path:
        return self.cache_dir / f"{symbol}_{period}_{interval}.csv"

    def _read_cache(
        self,
        symbol: str,
        period: str,
        interval: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(symbol, period, interval)
        if not self.cache or force_refresh or not cache_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        logger.debug("Cache hit: %s", symbol)
        return df

    def _write_cache(
        self,
        symbol: str,
        period: str,
        interval: str,
        df: pd.DataFrame,
    ) -> None:
        if self.cache and not df.empty:
            df.to_csv(self._cache_path(symbol, period, interval))

    # ── 单股 ───────────────────────────────────────────────────
    def fetch_single(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        返回 OHLCV DataFrame，列名标准化为小写。
        period 可用: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
        """
        cached = self._read_cache(symbol, period, interval, force_refresh=force_refresh)
        if not cached.empty:
            return cached

        try:
            tk = yf.Ticker(symbol)
            df = tk.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                logger.warning("Empty data for %s", symbol)
                return pd.DataFrame()
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            self._write_cache(symbol, period, interval, df)
            return df
        except Exception as e:
            logger.error("Failed to fetch %s: %s", symbol, e)
            return pd.DataFrame()

    # ── 批量 ───────────────────────────────────────────────────
    def fetch_bulk(
        self,
        symbols: list[str],
        period: str = "2y",
        interval: str = "1d",
        batch_size: int = 50,
        sleep: float = 1.0,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        批量拉取，按 batch_size 分组避免请求过大。
        返回 {symbol: DataFrame} 字典，失败标的自动跳过。
        """
        result: dict[str, pd.DataFrame] = {}
        pending: list[str] = []

        for symbol in symbols:
            cached = self._read_cache(symbol, period, interval, force_refresh=force_refresh)
            if not cached.empty:
                result[symbol] = cached
            else:
                pending.append(symbol)

        batches = [pending[i:i+batch_size] for i in range(0, len(pending), batch_size)]

        for batch in batches:
            try:
                raw = yf.download(
                    batch,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                )
                if raw.empty:
                    continue

                # 单只股票时 yfinance 不加 ticker 层级
                if len(batch) == 1:
                    df = raw.copy()
                    df.columns = [c.lower() for c in df.columns]
                    df.index.name = "date"
                    if not df.dropna().empty:
                        result[batch[0]] = df
                        self._write_cache(batch[0], period, interval, df)
                else:
                    for sym in batch:
                        try:
                            df = raw[sym].copy().dropna()
                            if df.empty:
                                continue
                            df.columns = [c.lower() for c in df.columns]
                            df.index.name = "date"
                            result[sym] = df
                            self._write_cache(sym, period, interval, df)
                        except KeyError:
                            logger.warning("No data returned for %s", sym)
            except Exception as e:
                logger.error("Batch download error: %s", e)
            time.sleep(sleep)

        logger.info(
            "Fetched %d / %d symbols (%d cached, %d downloaded)",
            len(result),
            len(symbols),
            len(result) - len(pending),
            len(pending),
        )
        return result

    # ── ETF Regime 快捷方式 ────────────────────────────────────
    def fetch_regime_etfs(
        self, etfs: Optional[list[str]] = None, period: str = "1y"
    ) -> dict[str, pd.DataFrame]:
        if etfs is None:
            etfs = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLV", "XLE",
                    "XLF", "SMH", "ARKK"]
        return self.fetch_bulk(etfs, period=period)

    # ── 最新收盘价字典（轻量查询）─────────────────────────────
    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        data = self.fetch_bulk(symbols, period="5d")
        return {sym: df["close"].iloc[-1] for sym, df in data.items() if not df.empty}
