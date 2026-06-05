from __future__ import annotations


def render_markdown_report(payload: dict) -> str:
    lines = [
        "# Daily Push Report",
        "",
        f"- Market Label: `{payload['market_summary']['final_label']}`",
        f"- Macro Score: `{payload['market_summary']['macro_score']}`",
        f"- Market Score: `{payload['market_summary']['market_score']}`",
        "",
        "## Final Score Table",
        "",
        "| Symbol | Final Score | Community | Flow | Event | Derivatives | Technical |",
        "|--------|-------------|-----------|------|-------|-------------|-----------|",
    ]
    for row in payload["symbols"]:
        futu = row["community"].get("futu", {})
        bullish_ratio = float(futu.get("bullish_ratio", 0.0))
        lines.append(
            f"| {row['display_name']} ({row['symbol']}) | {row['final_score']} | "
            f"{row['community_score']} / bullish {bullish_ratio*100:.1f}% | "
            f"{row['flow_score']} | {row['event_score']} | {row['derivatives_score']} | {row['technical_score']} |"
        )
    return "\n".join(lines)
