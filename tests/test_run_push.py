import unittest

from pipeline.run_push import build_delivery_payload


class RunPushTests(unittest.TestCase):
    def test_build_delivery_payload_includes_scan_and_watchlist_symbols(self) -> None:
        scan_output = {
            "summary": {"macro": {"score": 72}, "market_score": 66},
            "watchlist": [{"symbol": "US.NVDA", "final_score": 78, "action": "BUY_CANDIDATE", "technical_score": 63}],
        }
        watchlist_cfg = {
            "watchlist": [{"symbol": "US.TSLA", "alias": "Tesla", "enabled": True}]
        }
        merged_rows = {
            "US.NVDA": {"community": {"merged_score": 64, "futu": {}, "moomoo": {}}, "event": {"merged_score": 58}, "flow": {"merged_score": 60}, "derivatives": {"merged_score": 51}},
            "US.TSLA": {"community": {"merged_score": 62, "futu": {}, "moomoo": {}}, "event": {"merged_score": 57}, "flow": {"merged_score": 59}, "derivatives": {"merged_score": 50}},
        }

        payload = build_delivery_payload(scan_output, watchlist_cfg, merged_rows)

        self.assertEqual(payload["market_summary"]["macro_score"], 72)
        self.assertEqual(len(payload["symbols"]), 2)
        self.assertEqual(payload["symbols"][1]["display_name"], "Tesla")
