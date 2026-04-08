"""
adapters/news_client.py
─────────────────────────────────────────────────────────────
新闻标题适配器（RSS 优先，Finnhub 公司新闻备用）
- 不依赖 NewsAPI 付费层
- RSS 完全免费，无 key
- Finnhub 免费层按公司拉取，省配额

pip install feedparser requests
"""

from __future__ import annotations
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── RSS 免费新闻源 ─────────────────────────────────────────────
RSS_FEEDS = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "reuters_tech":     "https://feeds.reuters.com/news/technology",
    "ap_business":      "https://rsshub.app/apnews/topics/business",
    "cnbc_top":         "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "wsj_markets":      "https://feeds.content.dowjones.io/public/rss/mktwnews",
    "ft_markets":       "https://www.ft.com/rss/home/us",
}


class NewsClient:
    """新闻标题适配器"""

    def __init__(self, finnhub_key: str | None = None):
        self.finnhub_key = finnhub_key or os.environ.get("FINNHUB_API_KEY", "")

    # ── RSS 全局新闻 ───────────────────────────────────────────
    def fetch_rss_headlines(
        self,
        max_per_feed: int = 30,
        hours_back: int = 24,
    ) -> pd.DataFrame:
        """
        从 RSS 拉取最近 N 小时的财经新闻标题。
        返回 DataFrame: source | title | published | link
        """
        try:
            import feedparser
        except ImportError:
            raise ImportError("pip install feedparser")

        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours_back)
        rows: list[dict] = []

        for source, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_per_feed]:
                    # 解析时间
                    pub = None
                    for attr in ["published_parsed", "updated_parsed"]:
                        if hasattr(entry, attr) and getattr(entry, attr):
                            import calendar
                            ts = calendar.timegm(getattr(entry, attr))
                            pub = datetime.fromtimestamp(ts, tz=timezone.utc)
                            break
                    if pub and pub < cutoff:
                        continue
                    rows.append({
                        "source":    source,
                        "title":     entry.get("title", ""),
                        "published": pub.isoformat() if pub else "",
                        "link":      entry.get("link", ""),
                    })
            except Exception as e:
                logger.warning("RSS feed failed (%s): %s", source, e)

        if not rows:
            return pd.DataFrame(columns=["source","title","published","link"])
        df = pd.DataFrame(rows).drop_duplicates(subset=["title"])
        return df.sort_values("published", ascending=False).reset_index(drop=True)

    # ── Finnhub 公司新闻（备用，按标的拉）─────────────────────
    def fetch_company_news(
        self,
        symbol: str,
        days: int = 3,
    ) -> pd.DataFrame:
        """
        用 Finnhub 免费层拉取单只股票的公司新闻。
        按公司拉取比全市场扫描省配额。
        """
        if not self.finnhub_key:
            logger.warning("No Finnhub key, skipping company news for %s", symbol)
            return pd.DataFrame()

        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={symbol}&from={start}&to={end}&token={self.finnhub_key}"
        )
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)[["datetime","headline","source","url"]]
            df["published"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
            df = df.rename(columns={"headline":"title"})
            return df[["source","title","published","url"]].sort_values(
                "published", ascending=False
            ).reset_index(drop=True)
        except Exception as e:
            logger.warning("Finnhub company news failed (%s): %s", symbol, e)
            return pd.DataFrame()

    # ── 简单关键词情绪打分（规则型，不依赖 NLP 模型）──────────
    def score_headline_sentiment(self, headlines: pd.DataFrame) -> float:
        """
        对一批标题做简单关键词情绪打分。
        返回 -5 ~ +5 的浮点数，用于 L4 Sentiment 轻微加减分。
        """
        if headlines.empty or "title" not in headlines.columns:
            return 0.0

        BULL_WORDS = [
            "surge", "rally", "beat", "record", "strong", "bullish",
            "breakout", "upgraded", "buy", "outperform", "growth",
            "positive", "rise", "gain", "recovery", "momentum",
        ]
        BEAR_WORDS = [
            "crash", "plunge", "miss", "layoff", "recession", "bearish",
            "downgrade", "sell", "underperform", "weak", "concern",
            "decline", "fall", "risk", "warning", "cut", "loss",
        ]

        titles = headlines["title"].str.lower().fillna("")
        bull_count = titles.apply(
            lambda t: sum(w in t for w in BULL_WORDS)
        ).sum()
        bear_count = titles.apply(
            lambda t: sum(w in t for w in BEAR_WORDS)
        ).sum()

        total = bull_count + bear_count
        if total == 0:
            return 0.0

        ratio = (bull_count - bear_count) / total  # -1 ~ +1
        score = round(ratio * 5, 2)                # 缩放到 -5 ~ +5
        return max(-5.0, min(5.0, score))
