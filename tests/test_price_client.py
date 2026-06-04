import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.adapters.price_client import PriceClient


def _sample_price_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    frame = pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0, 11.5],
            "high": [10.5, 11.0, 11.5, 12.0],
            "low": [9.5, 10.0, 10.5, 11.0],
            "close": [10.2, 10.8, 11.2, 11.8],
            "volume": [1000, 1100, 1200, 1300],
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


class PriceClientTests(unittest.TestCase):
    def test_fetch_bulk_uses_cache_until_force_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cached = _sample_price_frame()
            cached.to_csv(cache_dir / "SPY_1y_1d.csv")

            client = PriceClient(cache=True, cache_dir=cache_dir)

            with patch("src.adapters.price_client.yf.download") as download:
                first = client.fetch_bulk(["SPY"], period="1y", force_refresh=False)
                self.assertFalse(download.called)
                self.assertEqual(first["SPY"]["close"].iloc[-1], 11.8)

            downloaded = cached.rename(columns=str.title)
            with patch("src.adapters.price_client.yf.download", return_value=downloaded) as download:
                second = client.fetch_bulk(["SPY"], period="1y", force_refresh=True)
                self.assertTrue(download.called)
                self.assertEqual(second["SPY"]["close"].iloc[-1], 11.8)
