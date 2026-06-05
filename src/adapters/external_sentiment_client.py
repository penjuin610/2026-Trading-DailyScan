from __future__ import annotations

import json
from pathlib import Path


def _resolve(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else base_dir / path


def load_sentiment_snapshots(config: dict, base_dir: Path) -> list[dict]:
    rows: list[dict] = []
    paths = config.get("external_inputs", {}).get("sentiment", {})
    for platform, path_str in paths.items():
        path = _resolve(base_dir, path_str)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            rows.append(
                {
                    "platform": platform,
                    "symbol": str(item.get("symbol", "")).upper(),
                    "bullish_ratio": float(item.get("bullish_ratio", 0.0)),
                    "neutral_ratio": float(item.get("neutral_ratio", 0.0)),
                    "bearish_ratio": float(item.get("bearish_ratio", 0.0)),
                    "score": float(item.get("score", 50.0)),
                }
            )
    return rows
