import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_rs_vs_sector_threshold_is_decimal_percentage(self) -> None:
        with open(ROOT / "config" / "thresholds.yaml", encoding="utf-8") as handle:
            thresholds = yaml.safe_load(handle)

        self.assertEqual(thresholds["structure"]["rs_vs_sector_min"], -0.02)
