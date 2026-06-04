import unittest

import pandas as pd

from src.features.community import compute_community_sentiment
from src.features.flow import compute_flow_signal
from src.features.scoring import compute_composite_score
from src.features.stock_events import compute_stock_event_signal


class CommunityAndScoringTests(unittest.TestCase):
    def test_community_sentiment_combines_futu_and_moomoo(self) -> None:
        posts = pd.DataFrame(
            [
                {"symbol": "MARKET", "source": "futu", "sentiment": "bullish", "mentions": 40},
                {"symbol": "MARKET", "source": "moomoo", "sentiment": "bullish", "mentions": 20},
                {"symbol": "MARKET", "source": "futu", "sentiment": "neutral", "mentions": 10},
                {"symbol": "MARKET", "source": "moomoo", "sentiment": "bearish", "mentions": 5},
                {"symbol": "NVDA", "source": "futu", "sentiment": "bullish", "mentions": 30},
                {"symbol": "NVDA", "source": "moomoo", "sentiment": "bearish", "mentions": 10},
            ]
        )

        result = compute_community_sentiment(posts, symbol="NVDA")

        self.assertEqual(result.market_label, "BULLISH")
        self.assertEqual(result.symbol_label, "BULLISH")
        self.assertGreater(result.market_score, 60)
        self.assertGreater(result.symbol_score, 55)

    def test_composite_score_uses_macro_led_weights(self) -> None:
        result = compute_composite_score(
            macro_score=90,
            market_score=80,
            community_score=70,
            flow_score=60,
            event_score=85,
            technical_score=40,
            cfg={
                "weights": {
                    "macro": 0.25,
                    "market": 0.20,
                    "community": 0.20,
                    "flow": 0.15,
                    "event": 0.15,
                    "technical": 0.05,
                }
            },
        )

        self.assertAlmostEqual(result.score, 76.25, places=2)
        self.assertEqual(result.label, "BUY_CANDIDATE")

    def test_flow_signal_combines_futu_and_moomoo_snapshots(self) -> None:
        flows = pd.DataFrame(
            [
                {"symbol": "NVDA", "source": "futu", "net_inflow": 2_400_000, "anomaly": True},
                {"symbol": "NVDA", "source": "moomoo", "net_inflow": 1_600_000, "anomaly": False},
                {"symbol": "NVDA", "source": "futu", "net_inflow": -200_000, "anomaly": False},
            ]
        )

        result = compute_flow_signal(flows, symbol="NVDA")

        self.assertGreater(result.score, 60)
        self.assertEqual(result.label, "BULLISH")
        self.assertGreater(result.net_inflow, 3_000_000)

    def test_stock_event_signal_detects_bearish_company_news(self) -> None:
        headlines = pd.DataFrame(
            [
                {"symbol": "AAPL", "title": "Apple cuts guidance after weak demand warning"},
                {"symbol": "AAPL", "title": "Analysts downgrade Apple on slowing iPhone trends"},
            ]
        )

        result = compute_stock_event_signal(headlines, symbol="AAPL")

        self.assertLess(result.score, 45)
        self.assertEqual(result.label, "BEARISH")
