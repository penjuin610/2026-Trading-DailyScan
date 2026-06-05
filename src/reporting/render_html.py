from __future__ import annotations


def _score_bar(label: str, value: float, css_class: str) -> str:
    return (
        f'<div class="metric-row"><span>{label}</span>'
        f'<div class="score-shell"><div class="score-bar {css_class}" style="width:{value}%">{value}</div></div></div>'
    )


def render_html_report(payload: dict) -> str:
    cards: list[str] = []
    for row in payload["symbols"]:
        cards.append(
            "\n".join(
                [
                    '<div class="symbol-card">',
                    f"<h2>{row['display_name']} ({row['symbol']})</h2>",
                    f'<div class="metric-row final-score">Final Score: {row["final_score"]}</div>',
                    _score_bar("Community", row["community_score"], "community-score"),
                    _score_bar("Flow", row["flow_score"], "flow-score"),
                    _score_bar("Event", row["event_score"], "event-score"),
                    _score_bar("Derivatives", row["derivatives_score"], "derivatives-score"),
                    _score_bar("Technical", row["technical_score"], "technical-score"),
                    "</div>",
                ]
            )
        )
    return """
<html>
  <body>
    %s
  </body>
</html>
""" % "\n".join(cards)
