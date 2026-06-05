from __future__ import annotations

import json
from pathlib import Path
import sys
import os

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapters.external_anomaly_client import load_derivatives_snapshots, load_flow_snapshots
from adapters.external_event_client import load_event_snapshots
from adapters.external_sentiment_client import load_sentiment_snapshots
from features.external_merge import merge_external_snapshot
from features.watchlist_merge import build_tracked_symbols
import pipeline.run_daily as daily_scan
from reporting.publishers import publish_reports
from reporting.render_html import render_html_report
from reporting.render_markdown import render_markdown_report


REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PUSH_MARKDOWN_PATH = REPORT_DIR / "daily_push_latest.md"
PUSH_HTML_PATH = REPORT_DIR / "daily_push_latest.html"
PUSH_JSON_PATH = ROOT / "outputs" / "data" / "latest_snapshot.json"
PUSH_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_watchlist_config(path: Path | None = None) -> dict:
    config_path = path or (ROOT / "config" / "my_watchlist.yaml")
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _final_score(macro_score: float, market_score: float, community_score: float, flow_score: float, event_score: float, derivatives_score: float, technical_score: float) -> float:
    return round(
        macro_score * 0.22
        + market_score * 0.18
        + community_score * 0.18
        + flow_score * 0.14
        + event_score * 0.14
        + derivatives_score * 0.09
        + technical_score * 0.05,
        2,
    )


def build_delivery_payload(scan_output: dict, watchlist_cfg: dict, merged_rows: dict[str, dict]) -> dict:
    tracked = build_tracked_symbols(scan_output.get("watchlist", []), watchlist_cfg)
    scan_by_symbol = {row["symbol"]: row for row in scan_output.get("watchlist", [])}
    macro_score = float(scan_output["summary"]["macro"]["score"])
    market_score = float(scan_output["summary"]["market_score"])
    symbols = []

    for item in tracked:
        symbol = item["symbol"]
        scan_row = scan_by_symbol.get(symbol, {})
        merged = merged_rows.get(
            symbol,
            {
                "community": {"merged_score": 50.0, "futu": {}, "moomoo": {}},
                "event": {"merged_score": 50.0, "futu": {}, "moomoo": {}},
                "flow": {"merged_score": 50.0, "futu": {}, "moomoo": {}},
                "derivatives": {"merged_score": 50.0, "futu": {}, "moomoo": {}},
            },
        )
        technical_score = float(scan_row.get("technical_score", scan_row.get("trigger_score", 50.0)))
        final_score = _final_score(
            macro_score=macro_score,
            market_score=market_score,
            community_score=float(merged["community"]["merged_score"]),
            flow_score=float(merged["flow"]["merged_score"]),
            event_score=float(merged["event"]["merged_score"]),
            derivatives_score=float(merged["derivatives"]["merged_score"]),
            technical_score=technical_score,
        )
        if final_score >= 70:
            final_label = "BULLISH"
        elif final_score <= 45:
            final_label = "BEARISH"
        else:
            final_label = "NEUTRAL"
        symbols.append(
            {
                "symbol": symbol,
                "display_name": item.get("alias", symbol),
                "source": item["source"],
                "final_score": final_score,
                "final_label": final_label,
                "community_score": float(merged["community"]["merged_score"]),
                "flow_score": float(merged["flow"]["merged_score"]),
                "event_score": float(merged["event"]["merged_score"]),
                "derivatives_score": float(merged["derivatives"]["merged_score"]),
                "technical_score": technical_score,
                "community": merged["community"],
                "event": merged["event"],
                "flow": merged["flow"],
                "derivatives": merged["derivatives"],
                "summary_line": f"{symbol}: {final_label} | final={final_score:.1f} | community={merged['community']['merged_score']:.1f} | flow={merged['flow']['merged_score']:.1f}",
            }
        )

    symbols.sort(key=lambda row: row["final_score"], reverse=True)
    return {
        "market_summary": {
            "macro_score": macro_score,
            "market_score": market_score,
            "final_label": "RISK_ON" if macro_score >= 60 and market_score >= 60 else "NEUTRAL",
        },
        "symbols": symbols,
    }


def run_push() -> dict:
    watchlist_cfg = load_watchlist_config()
    os.environ["DAILY_SCAN_OUTPUT_DIR"] = str(REPORT_DIR)
    daily_scan.OUTPUT_DIR = REPORT_DIR
    daily_scan.LATEST_REPORT_PATH = REPORT_DIR / "daily_scan_latest.md"
    scan_output = daily_scan.run_pipeline()
    base_dir = ROOT
    sentiment_rows = load_sentiment_snapshots(watchlist_cfg, base_dir)
    event_rows = load_event_snapshots(watchlist_cfg, base_dir)
    flow_rows = load_flow_snapshots(watchlist_cfg, base_dir)
    derivatives_rows = load_derivatives_snapshots(watchlist_cfg, base_dir)

    tracked = build_tracked_symbols(scan_output.get("watchlist", []), watchlist_cfg)
    merged_rows = {
        item["symbol"]: merge_external_snapshot(
            symbol=item["symbol"],
            sentiment_rows=sentiment_rows,
            event_rows=event_rows,
            flow_rows=flow_rows,
            derivatives_rows=derivatives_rows,
        )
        for item in tracked
    }

    payload = build_delivery_payload(scan_output, watchlist_cfg, merged_rows)
    markdown = render_markdown_report(payload)
    html = render_html_report(payload)

    PUSH_MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
    PUSH_HTML_PATH.write_text(html, encoding="utf-8")
    PUSH_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    top_symbols = ", ".join(f"{row['display_name']} {row['final_score']:.1f}" for row in payload["symbols"][:3])
    summary_text = f"{payload['market_summary']['final_label']} | Top: {top_symbols}" if top_symbols else payload["market_summary"]["final_label"]
    publish_result = publish_reports(watchlist_cfg, str(PUSH_MARKDOWN_PATH), str(PUSH_HTML_PATH), summary_text)
    payload["publish_result"] = publish_result
    return payload


if __name__ == "__main__":
    result = run_push()
    print(f"Watchlist push complete: {len(result['symbols'])} symbols")
