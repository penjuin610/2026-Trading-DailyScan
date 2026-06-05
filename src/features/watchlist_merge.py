from __future__ import annotations


def build_tracked_symbols(scan_rows: list[dict], watchlist_cfg: dict) -> list[dict]:
    tracked: list[dict] = []
    seen: dict[str, dict] = {}

    for row in scan_rows:
        symbol = str(row["symbol"]).upper()
        if symbol in seen:
            continue
        item = {"symbol": symbol, "source": "scan"}
        seen[symbol] = item
        tracked.append(item)

    for entry in watchlist_cfg.get("watchlist", []):
        if not entry.get("enabled", True):
            continue
        symbol = str(entry["symbol"]).upper()
        alias = entry.get("alias")
        if symbol in seen:
            seen[symbol]["source"] = "scan+watchlist"
            if alias:
                seen[symbol]["alias"] = alias
            continue
        item = {"symbol": symbol, "source": "watchlist"}
        if alias:
            item["alias"] = alias
        seen[symbol] = item
        tracked.append(item)

    return tracked
