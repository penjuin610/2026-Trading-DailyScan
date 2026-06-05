import unittest

from src.features.watchlist_merge import build_tracked_symbols


class WatchlistMergeTests(unittest.TestCase):
    def test_build_tracked_symbols_keeps_scan_symbols_and_enabled_watchlist(self) -> None:
        scan_rows = [{"symbol": "US.NVDA"}, {"symbol": "US.TSLA"}]
        watchlist_cfg = {
            "watchlist": [
                {"symbol": "US.TSLA", "alias": "Tesla", "enabled": True},
                {"symbol": "US.PDD", "alias": "PDD", "enabled": True},
                {"symbol": "US.FIG", "alias": "Figma", "enabled": False},
            ]
        }

        tracked = build_tracked_symbols(scan_rows, watchlist_cfg)

        self.assertEqual(
            tracked,
            [
                {"symbol": "US.NVDA", "source": "scan"},
                {"symbol": "US.TSLA", "source": "scan+watchlist", "alias": "Tesla"},
                {"symbol": "US.PDD", "source": "watchlist", "alias": "PDD"},
            ],
        )
