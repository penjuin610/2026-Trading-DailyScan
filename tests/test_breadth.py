import unittest

import numpy as np
import pandas as pd

from src.features.breadth import compute_breadth


def _trend_with_last_day_pullback(start: float, finish: float, periods: int = 220) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    closes = np.linspace(start, finish, periods)
    closes[-2] = finish + 5
    closes[-1] = finish - 5
    frame = pd.DataFrame(
        {
            "close": closes,
            "open": closes - 0.5,
            "high": closes + 1,
            "low": closes - 1,
            "volume": np.linspace(1000, 1500, periods),
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


class BreadthTests(unittest.TestCase):
    def test_breadth_thresholds_respect_adv_ratio_and_nh_nl_minimums(self) -> None:
        universe_prices = {
            "AAA": _trend_with_last_day_pullback(100, 180),
            "BBB": _trend_with_last_day_pullback(60, 140),
        }
        etf_prices = {
            "SPY": _trend_with_last_day_pullback(300, 450),
            "XLK": _trend_with_last_day_pullback(150, 250),
        }

        result = compute_breadth(
            universe_prices=universe_prices,
            etf_prices=etf_prices,
            sector_etfs=["XLK"],
            cfg={
                "score_green": 60,
                "score_yellow": 40,
                "pct_above_200ema_green": 0.55,
                "pct_above_200ema_yellow": 0.40,
                "adv_ratio_min": 0.50,
                "nh_nl_ratio_min": 1.20,
            },
        )

        self.assertNotEqual(result.level, "HEALTHY")
        self.assertEqual(result.adv_ratio, 0.0)
        self.assertLess(result.nh_nl_ratio, 1.20)
