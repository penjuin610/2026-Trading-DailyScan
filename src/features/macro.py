from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .event_risk import EventRisk


@dataclass
class MacroContextResult:
    score: float
    label: str
    gate_pass: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "label": self.label,
            "gate_pass": self.gate_pass,
            "note": self.note,
        }


def compute_macro_context(
    macro_df: pd.DataFrame | None,
    event_risk: EventRisk,
    cfg: dict | None = None,
) -> MacroContextResult:
    cfg = cfg or {}
    score = 55.0
    notes: list[str] = []
    gate_pass = event_risk.gate_pass

    if event_risk.level == "BLOCK":
        score -= float(cfg.get("score_event_block_penalty", 20))
        notes.append("High-impact macro event inside block window")
    elif event_risk.level == "WARNING":
        score -= 8.0
        notes.append("Major macro event approaching")

    if macro_df is not None and not macro_df.empty:
        last = macro_df.iloc[-1]
        us10y_bps = float(last.get("us10y_5d_chg", 0.0) * 100)
        dxy_pct = float(last.get("dxy_pct5d", 0.0))
        oil_pct = float(last.get("oil_wti_pct5d", 0.0))

        if us10y_bps >= float(cfg.get("us10y_5d_warn_bps", 15)):
            score -= 10
            notes.append(f"Rates pressure ({us10y_bps:+.1f} bps / 5d)")
        elif us10y_bps <= -10:
            score += 5

        if dxy_pct >= float(cfg.get("dxy_5d_warn_pct", 0.015)):
            score -= 8
            notes.append(f"USD risk-off ({dxy_pct*100:+.1f}% / 5d)")
        elif dxy_pct <= -0.01:
            score += 4

        if oil_pct >= float(cfg.get("oil_5d_warn_pct", 0.05)):
            score -= 4
            notes.append(f"Oil shock ({oil_pct*100:+.1f}% / 5d)")

    score = max(0.0, min(100.0, score))
    bearish_min = float(cfg.get("bearish_min_score", 45))
    bullish_min = float(cfg.get("bullish_min_score", 60))
    if score >= bullish_min:
        label = "BULLISH"
    elif score <= bearish_min:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return MacroContextResult(
        score=round(score, 2),
        label=label,
        gate_pass=gate_pass and score > bearish_min,
        note=" | ".join(notes) if notes else f"Macro {label}",
    )
