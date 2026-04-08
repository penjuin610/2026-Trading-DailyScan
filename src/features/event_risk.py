"""
features/event_risk.py  —  L1 Event Risk
─────────────────────────────────────────────────────────────
识别未来 24/48h 内的高影响宏观事件，决定是否禁止新开仓。

输出：
  EventRisk(
    gate_pass  = True/False,  # False = 禁止新开仓
    level      = "CLEAR" / "WARNING" / "BLOCK",
    events_24h = [...],       # 未来24h内的事件列表
    events_48h = [...],
    note       = "说明文字"
  )
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CALENDAR_PATH = Path(__file__).parents[2] / "config" / "events_calendar.csv"


@dataclass
class EventRisk:
    gate_pass:  bool
    level:      str          # CLEAR / WARNING / BLOCK
    events_24h: list[dict] = field(default_factory=list)
    events_48h: list[dict] = field(default_factory=list)
    note:       str = ""

    def to_dict(self) -> dict:
        return {
            "gate_pass":       self.gate_pass,
            "level":           self.level,
            "events_24h_count": len(self.events_24h),
            "events_48h_count": len(self.events_48h),
            "events_24h":      self.events_24h,
            "note":            self.note,
        }


def check_event_risk(
    calendar_path: Path = CALENDAR_PATH,
    now: datetime | None = None,
    block_window_hours: int = 24,
    warn_window_hours:  int = 48,
) -> EventRisk:
    """
    读取事件日历，判断当前时间点的风险等级。

    规则：
    - 未来 24h 内有 HIGH 事件 → BLOCK（gate_pass=False）
    - 未来 24-48h 有 HIGH 事件 → WARNING（gate_pass=True 但附注）
    - 否则 → CLEAR
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    # 加载日历
    if not calendar_path.exists():
        logger.warning("Events calendar not found: %s", calendar_path)
        return EventRisk(gate_pass=True, level="CLEAR", note="No calendar file")

    try:
        cal = pd.read_csv(calendar_path, parse_dates=["date"])
    except Exception as e:
        logger.error("Failed to load events calendar: %s", e)
        return EventRisk(gate_pass=True, level="CLEAR", note=f"Calendar load error: {e}")

    # 时间窗口
    t_24h = now + timedelta(hours=block_window_hours)
    t_48h = now + timedelta(hours=warn_window_hours)

    # 确保时区一致
    cal["date"] = pd.to_datetime(cal["date"]).dt.tz_localize("UTC", ambiguous="infer",
                                                               nonexistent="shift_forward")

    now_ts = now
    events_24h = cal[
        (cal["date"] >= now_ts) &
        (cal["date"] <= t_24h) &
        (cal["impact"].str.upper() == "HIGH")
    ].to_dict("records")

    events_48h = cal[
        (cal["date"] > t_24h) &
        (cal["date"] <= t_48h) &
        (cal["impact"].str.upper() == "HIGH")
    ].to_dict("records")

    if events_24h:
        event_names = ", ".join(e["event"] for e in events_24h)
        return EventRisk(
            gate_pass=False,
            level="BLOCK",
            events_24h=events_24h,
            events_48h=events_48h,
            note=f"BLOCK: High-impact event within 24h — {event_names}",
        )
    elif events_48h:
        event_names = ", ".join(e["event"] for e in events_48h)
        return EventRisk(
            gate_pass=True,
            level="WARNING",
            events_24h=[],
            events_48h=events_48h,
            note=f"WARNING: High-impact event within 48h — {event_names}. Reduce position size.",
        )
    else:
        return EventRisk(
            gate_pass=True,
            level="CLEAR",
            note="No high-impact events in next 48h",
        )
