import unittest

from src.features.external_merge import merge_external_snapshot


class ExternalMergeTests(unittest.TestCase):
    def test_merge_external_snapshot_keeps_platform_ratios_and_scores(self) -> None:
        merged = merge_external_snapshot(
            symbol="US.TSLA",
            sentiment_rows=[
                {"platform": "futu", "symbol": "US.TSLA", "bullish_ratio": 0.60, "neutral_ratio": 0.25, "bearish_ratio": 0.15, "score": 68},
                {"platform": "moomoo", "symbol": "US.TSLA", "bullish_ratio": 0.50, "neutral_ratio": 0.30, "bearish_ratio": 0.20, "score": 60},
            ],
            event_rows=[
                {"platform": "futu", "symbol": "US.TSLA", "label": "BULLISH", "score": 62, "summary": "delivery beat"},
                {"platform": "moomoo", "symbol": "US.TSLA", "label": "NEUTRAL", "score": 55, "summary": "mixed take"},
            ],
            flow_rows=[
                {"platform": "futu", "symbol": "US.TSLA", "label": "BULLISH", "score": 64},
                {"platform": "moomoo", "symbol": "US.TSLA", "label": "BULLISH", "score": 58},
            ],
            derivatives_rows=[
                {"platform": "futu", "symbol": "US.TSLA", "label": "BULLISH", "score": 57},
                {"platform": "moomoo", "symbol": "US.TSLA", "label": "BEARISH", "score": 46},
            ],
        )

        self.assertEqual(merged["community"]["futu"]["bullish_ratio"], 0.60)
        self.assertEqual(merged["community"]["moomoo"]["bearish_ratio"], 0.20)
        self.assertAlmostEqual(merged["community"]["merged_score"], 64.0)
        self.assertAlmostEqual(merged["flow"]["merged_score"], 61.0)
        self.assertAlmostEqual(merged["event"]["merged_score"], 58.5)
