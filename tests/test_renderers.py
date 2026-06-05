import unittest

from src.reporting.render_html import render_html_report
from src.reporting.render_markdown import render_markdown_report


class RendererTests(unittest.TestCase):
    def test_renderers_include_watchlist_rows_and_visual_scores(self) -> None:
        payload = {
            "market_summary": {"macro_score": 72, "market_score": 66, "final_label": "RISK_ON"},
            "symbols": [
                {
                    "symbol": "US.TSLA",
                    "display_name": "Tesla",
                    "final_score": 67,
                    "community_score": 64,
                    "flow_score": 61,
                    "event_score": 58.5,
                    "derivatives_score": 51.5,
                    "technical_score": 49,
                    "community": {
                        "futu": {"bullish_ratio": 0.60, "neutral_ratio": 0.25, "bearish_ratio": 0.15},
                        "moomoo": {"bullish_ratio": 0.50, "neutral_ratio": 0.30, "bearish_ratio": 0.20},
                    },
                }
            ],
        }

        markdown = render_markdown_report(payload)
        html = render_html_report(payload)

        self.assertIn("Tesla", markdown)
        self.assertIn("Final Score", markdown)
        self.assertIn("60.0%", markdown)
        self.assertIn("community-score", html)
        self.assertIn("TSLA", html)
