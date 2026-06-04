from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompositeScoreResult:
    score: float
    label: str
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "label": self.label,
            "components": {key: round(val, 2) for key, val in self.components.items()},
        }


DEFAULT_WEIGHTS = {
    "macro": 0.25,
    "market": 0.20,
    "community": 0.20,
    "flow": 0.15,
    "event": 0.15,
    "technical": 0.05,
}


def compute_composite_score(
    macro_score: float,
    market_score: float,
    community_score: float,
    flow_score: float,
    event_score: float,
    technical_score: float,
    cfg: dict | None = None,
) -> CompositeScoreResult:
    cfg = cfg or {}
    weights = {**DEFAULT_WEIGHTS, **cfg.get("weights", {})}
    total_weight = sum(weights.values()) or 1.0
    normalized = {key: value / total_weight for key, value in weights.items()}

    components = {
        "macro": macro_score,
        "market": market_score,
        "community": community_score,
        "flow": flow_score,
        "event": event_score,
        "technical": technical_score,
    }
    score = sum(components[key] * normalized[key] for key in components)
    buy_min = float(cfg.get("buy_candidate_min", 75))
    watch_min = float(cfg.get("watch_min", 60))

    if score >= buy_min:
        label = "BUY_CANDIDATE"
    elif score >= watch_min:
        label = "WATCH"
    else:
        label = "PASS"

    return CompositeScoreResult(score=round(score, 2), label=label, components=components)
