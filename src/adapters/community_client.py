from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]


class CommunitySnapshotClient:
    """Load exported Futu/moomoo community sentiment snapshots."""

    def __init__(self, snapshot_paths: dict[str, str] | None = None):
        self.snapshot_paths = snapshot_paths or {}

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str).expanduser()
        return path if path.is_absolute() else ROOT / path

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "CommunitySnapshotClient":
        providers = (cfg or {}).get("providers", {})
        snapshot_paths = providers.get("community_snapshot_paths", {})
        env_overrides = {
            "futu": os.environ.get("FUTU_COMMUNITY_SNAPSHOT_PATH"),
            "moomoo": os.environ.get("MOOMOO_COMMUNITY_SNAPSHOT_PATH"),
        }
        merged = {**snapshot_paths, **{k: v for k, v in env_overrides.items() if v}}
        return cls(snapshot_paths=merged)

    def fetch(self) -> pd.DataFrame:
        rows: list[dict] = []
        for source, path_str in self.snapshot_paths.items():
            path = self._resolve_path(path_str)
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Community snapshot load failed (%s): %s", path, exc)
                continue

            if isinstance(payload, dict):
                payload = payload.get("items", [])

            for item in payload:
                rows.append(
                    {
                        "source": source,
                        "symbol": str(item.get("symbol", "MARKET")).upper(),
                        "sentiment": str(item.get("sentiment", "neutral")).lower(),
                        "mentions": float(item.get("mentions", item.get("count", 1))),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["source", "symbol", "sentiment", "mentions"])
        return pd.DataFrame(rows)
