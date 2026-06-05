from __future__ import annotations


def _platform_map(rows: list[dict], symbol: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        if str(row.get("symbol", "")).upper() == symbol.upper():
            result[str(row["platform"]).lower()] = row
    return result


def _mean_score(platform_rows: dict[str, dict], default: float = 50.0) -> float:
    scores = [float(row["score"]) for row in platform_rows.values() if "score" in row]
    return round(sum(scores) / len(scores), 2) if scores else default


def merge_external_snapshot(
    symbol: str,
    sentiment_rows: list[dict],
    event_rows: list[dict],
    flow_rows: list[dict],
    derivatives_rows: list[dict],
) -> dict:
    sentiment_map = _platform_map(sentiment_rows, symbol)
    event_map = _platform_map(event_rows, symbol)
    flow_map = _platform_map(flow_rows, symbol)
    derivatives_map = _platform_map(derivatives_rows, symbol)

    return {
        "community": {
            "futu": sentiment_map.get("futu", {}),
            "moomoo": sentiment_map.get("moomoo", {}),
            "merged_score": _mean_score(sentiment_map),
        },
        "event": {
            "futu": event_map.get("futu", {}),
            "moomoo": event_map.get("moomoo", {}),
            "merged_score": _mean_score(event_map),
        },
        "flow": {
            "futu": flow_map.get("futu", {}),
            "moomoo": flow_map.get("moomoo", {}),
            "merged_score": _mean_score(flow_map),
        },
        "derivatives": {
            "futu": derivatives_map.get("futu", {}),
            "moomoo": derivatives_map.get("moomoo", {}),
            "merged_score": _mean_score(derivatives_map),
        },
    }
