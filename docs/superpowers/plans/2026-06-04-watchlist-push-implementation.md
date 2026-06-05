# Watchlist Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-click local watchlist push workflow that combines the existing macro/market scan with dual-platform Futu+moomoo content signals, renders Markdown/HTML reports, and sends them to Discord and Email.

**Architecture:** Keep [pipeline/run_daily.py](/Users/xiao1/Documents/Xiao1 trading/pipeline/run_daily.py:1) as the market-scanning core, then add a new push orchestrator that loads a manual watchlist, collects dual-platform external signal snapshots, merges them into per-symbol scorecards, renders reports, and publishes notifications. Use file-based adapters first so the system works without OpenD and degrades gracefully when some external signals are unavailable.

**Tech Stack:** Python 3.11, `unittest`, YAML, standard-library HTML/email/webhook publishing, existing repository pipeline modules.

---

## File Structure

- Create: `config/my_watchlist.yaml`
  Responsibility: user-maintained watchlist, notification settings, report knobs.
- Create: `src/adapters/external_sentiment_client.py`
  Responsibility: load paired Futu/moomoo community sentiment snapshots from configured JSON paths and normalize them.
- Create: `src/adapters/external_event_client.py`
  Responsibility: load paired Futu/moomoo event/digest snapshots and normalize evidence records.
- Create: `src/adapters/external_anomaly_client.py`
  Responsibility: load paired flow/derivatives anomaly snapshots and normalize them.
- Create: `src/features/watchlist_merge.py`
  Responsibility: merge scan candidates and configured watchlist symbols into one deduplicated tracked universe.
- Create: `src/features/external_merge.py`
  Responsibility: convert normalized dual-platform external inputs into merged community/event/flow/derivatives scores.
- Create: `src/reporting/render_markdown.py`
  Responsibility: render the one-click delivery Markdown report.
- Create: `src/reporting/render_html.py`
  Responsibility: render the HTML report with simple score bars and ratio visualization.
- Create: `src/reporting/publishers.py`
  Responsibility: push report content to Discord webhook and SMTP email with failure isolation.
- Create: `pipeline/run_push.py`
  Responsibility: one-click orchestrator for config load, local scan, external merge, report render, and notification publish.
- Create: `run_daily_push.command`
  Responsibility: desktop double-click launcher for local use.
- Modify: `README.md`
  Responsibility: document English usage for one-click push, watchlist editing, and notifications.
- Modify: `README.zh-CN.md`
  Responsibility: document Chinese usage for one-click push, watchlist editing, and notifications.
- Test: `tests/test_watchlist_merge.py`
- Test: `tests/test_external_merge.py`
- Test: `tests/test_renderers.py`
- Test: `tests/test_publishers.py`
- Test: `tests/test_run_push.py`

### Task 1: Add Watchlist Config and Merge Logic

**Files:**
- Create: `config/my_watchlist.yaml`
- Create: `src/features/watchlist_merge.py`
- Test: `tests/test_watchlist_merge.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from src.features.watchlist_merge import build_tracked_symbols


class WatchlistMergeTests(unittest.TestCase):
    def test_build_tracked_symbols_keeps_scan_symbols_and_enabled_watchlist(self) -> None:
        scan_rows = [{"symbol": "US.NVDA"}, {"symbol": "US.TSLA"}]
        watchlist_cfg = {
            "watchlist": [
                {"symbol": "US.TSLA", "alias": "Tesla", "enabled": True},
                {"symbol": "US.PDD", "alias": "PDD", "enabled": True},
                {"symbol": "US.FIG", "alias": "Figma", "enabled": False},
            ]
        }

        tracked = build_tracked_symbols(scan_rows, watchlist_cfg)

        self.assertEqual(
            tracked,
            [
                {"symbol": "US.NVDA", "source": "scan"},
                {"symbol": "US.TSLA", "source": "scan+watchlist", "alias": "Tesla"},
                {"symbol": "US.PDD", "source": "watchlist", "alias": "PDD"},
            ],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_watchlist_merge -v`
Expected: `ModuleNotFoundError` or missing `build_tracked_symbols`

- [ ] **Step 3: Write minimal implementation**

```python
def build_tracked_symbols(scan_rows: list[dict], watchlist_cfg: dict) -> list[dict]:
    scan_symbols = []
    seen = {}
    for row in scan_rows:
        symbol = str(row["symbol"]).upper()
        if symbol not in seen:
            item = {"symbol": symbol, "source": "scan"}
            seen[symbol] = item
            scan_symbols.append(item)

    for item in watchlist_cfg.get("watchlist", []):
        if not item.get("enabled", True):
            continue
        symbol = str(item["symbol"]).upper()
        alias = item.get("alias")
        if symbol in seen:
            seen[symbol]["source"] = "scan+watchlist"
            if alias:
                seen[symbol]["alias"] = alias
            continue
        payload = {"symbol": symbol, "source": "watchlist"}
        if alias:
            payload["alias"] = alias
        seen[symbol] = payload
        scan_symbols.append(payload)
    return scan_symbols
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_watchlist_merge -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add config/my_watchlist.yaml src/features/watchlist_merge.py tests/test_watchlist_merge.py
git commit -m "Add watchlist config and merge logic"
```

### Task 2: Add Dual-Platform External Snapshot Adapters

**Files:**
- Create: `src/adapters/external_sentiment_client.py`
- Create: `src/adapters/external_event_client.py`
- Create: `src/adapters/external_anomaly_client.py`
- Test: `tests/test_external_merge.py`

- [ ] **Step 1: Write the failing adapter/merge test**

```python
import unittest

from src.features.external_merge import merge_external_snapshot


class ExternalMergeTests(unittest.TestCase):
    def test_merge_external_snapshot_keeps_platform_ratios_and_scores(self) -> None:
        merged = merge_external_snapshot(
            symbol="US.TSLA",
            sentiment_rows=[
                {"platform": "futu", "symbol": "US.TSLA", "bullish_ratio": 0.60, "neutral_ratio": 0.25, "bearish_ratio": 0.15, "score": 68},
                {"platform": "moomoo", "symbol": "US.TSLA", "bullish_ratio": 0.50, "neutral_ratio": 0.30, "bearish_ratio": 0.20, "score": 60},
            ],
            event_rows=[
                {"platform": "futu", "symbol": "US.TSLA", "label": "BULLISH", "score": 62, "summary": "delivery beat"},
                {"platform": "moomoo", "symbol": "US.TSLA", "label": "NEUTRAL", "score": 55, "summary": "mixed take"},
            ],
            flow_rows=[
                {"platform": "futu", "symbol": "US.TSLA", "label": "BULLISH", "score": 64},
                {"platform": "moomoo", "symbol": "US.TSLA", "label": "BULLISH", "score": 58},
            ],
            derivatives_rows=[
                {"platform": "futu", "symbol": "US.TSLA", "label": "BULLISH", "score": 57},
                {"platform": "moomoo", "symbol": "US.TSLA", "label": "BEARISH", "score": 46},
            ],
        )

        self.assertEqual(merged["community"]["futu"]["bullish_ratio"], 0.60)
        self.assertEqual(merged["community"]["moomoo"]["bearish_ratio"], 0.20)
        self.assertAlmostEqual(merged["community"]["merged_score"], 64.0)
        self.assertAlmostEqual(merged["flow"]["merged_score"], 61.0)
        self.assertAlmostEqual(merged["event"]["merged_score"], 58.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_external_merge -v`
Expected: missing module or missing function failure

- [ ] **Step 3: Write minimal implementation**

```python
def _platform_map(rows: list[dict], symbol: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if str(row.get("symbol", "")).upper() == symbol.upper():
            result[str(row["platform"]).lower()] = row
    return result


def _mean_score(platform_rows: dict[str, dict], default: float = 50.0) -> float:
    scores = [float(row["score"]) for row in platform_rows.values() if "score" in row]
    return round(sum(scores) / len(scores), 2) if scores else default


def merge_external_snapshot(symbol: str, sentiment_rows: list[dict], event_rows: list[dict], flow_rows: list[dict], derivatives_rows: list[dict]) -> dict:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_external_merge -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/adapters/external_sentiment_client.py src/adapters/external_event_client.py src/adapters/external_anomaly_client.py src/features/external_merge.py tests/test_external_merge.py
git commit -m "Add dual-platform external snapshot merge"
```

### Task 3: Add Report Renderers

**Files:**
- Create: `src/reporting/render_markdown.py`
- Create: `src/reporting/render_html.py`
- Test: `tests/test_renderers.py`

- [ ] **Step 1: Write the failing renderer test**

```python
import unittest

from src.reporting.render_markdown import render_markdown_report
from src.reporting.render_html import render_html_report


class RendererTests(unittest.TestCase):
    def test_renderers_include_watchlist_rows_and_visual_scores(self) -> None:
        payload = {
            "market_summary": {"macro_score": 72, "market_score": 66, "final_label": "RISK_ON"},
            "symbols": [
                {
                    "symbol": "US.TSLA",
                    "display_name": "Tesla",
                    "final_score": 67,
                    "community_score": 64,
                    "flow_score": 61,
                    "event_score": 58.5,
                    "derivatives_score": 51.5,
                    "technical_score": 49,
                    "community": {
                        "futu": {"bullish_ratio": 0.60, "neutral_ratio": 0.25, "bearish_ratio": 0.15},
                        "moomoo": {"bullish_ratio": 0.50, "neutral_ratio": 0.30, "bearish_ratio": 0.20},
                    },
                }
            ],
        }

        markdown = render_markdown_report(payload)
        html = render_html_report(payload)

        self.assertIn("Tesla", markdown)
        self.assertIn("Final Score", markdown)
        self.assertIn("60.0%", markdown)
        self.assertIn("community-score", html)
        self.assertIn("TSLA", html)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_renderers -v`
Expected: import failure

- [ ] **Step 3: Write minimal implementation**

```python
def render_markdown_report(payload: dict) -> str:
    lines = [
        "# Daily Push Report",
        "",
        f"- Market Label: `{payload['market_summary']['final_label']}`",
        f"- Macro Score: `{payload['market_summary']['macro_score']}`",
        f"- Market Score: `{payload['market_summary']['market_score']}`",
        "",
        "| Symbol | Final Score | Community | Flow | Event | Derivatives | Technical |",
        "|--------|-------------|-----------|------|-------|-------------|-----------|",
    ]
    for row in payload["symbols"]:
        futu = row["community"]["futu"]
        lines.append(
            f"| {row['display_name']} ({row['symbol']}) | {row['final_score']} | "
            f"{row['community_score']} / bullish {futu['bullish_ratio']*100:.1f}% | "
            f"{row['flow_score']} | {row['event_score']} | {row['derivatives_score']} | {row['technical_score']} |"
        )
    return "\n".join(lines)
```

```python
def render_html_report(payload: dict) -> str:
    cards = []
    for row in payload["symbols"]:
        cards.append(
            f\"\"\"<div class="symbol-card">
  <h2>{row['display_name']} ({row['symbol']})</h2>
  <div class="community-score">Final Score: {row['final_score']}</div>
  <div class="score-bar community-score" style="width:{row['community_score']}%">Community {row['community_score']}</div>
</div>\"\"\"
        )
    return "<html><body>" + "".join(cards) + "</body></html>"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_renderers -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/reporting/render_markdown.py src/reporting/render_html.py tests/test_renderers.py
git commit -m "Add watchlist push report renderers"
```

### Task 4: Add Discord and Email Publishers

**Files:**
- Create: `src/reporting/publishers.py`
- Test: `tests/test_publishers.py`

- [ ] **Step 1: Write the failing publisher test**

```python
import os
import tempfile
import unittest
from unittest.mock import patch

from src.reporting.publishers import publish_reports


class PublisherTests(unittest.TestCase):
    def test_publish_reports_keeps_email_running_when_discord_fails(self) -> None:
        os.environ["DISCORD_WEBHOOK_URL"] = "https://discord.test/webhook"
        os.environ["SMTP_HOST"] = "smtp.test"
        os.environ["SMTP_PORT"] = "587"
        os.environ["SMTP_USERNAME"] = "user"
        os.environ["SMTP_PASSWORD"] = "pass"
        os.environ["SMTP_FROM"] = "from@test"
        os.environ["SMTP_TO"] = "to@test"

        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = f"{tmpdir}/report.md"
            html_path = f"{tmpdir}/report.html"
            open(markdown_path, "w", encoding="utf-8").write("# report")
            open(html_path, "w", encoding="utf-8").write("<html></html>")

            with patch("src.reporting.publishers._post_discord", side_effect=RuntimeError("discord down")), patch("src.reporting.publishers._send_email") as send_email:
                result = publish_reports(
                    {"notifications": {"discord": {"enabled": True, "webhook_env": "DISCORD_WEBHOOK_URL"}, "email": {"enabled": True, "smtp_host_env": "SMTP_HOST", "smtp_port_env": "SMTP_PORT", "username_env": "SMTP_USERNAME", "password_env": "SMTP_PASSWORD", "from_env": "SMTP_FROM", "to_env": "SMTP_TO"}}},
                    markdown_path,
                    html_path,
                    "summary",
                )

        self.assertFalse(result["discord"]["ok"])
        self.assertTrue(result["email"]["ok"])
        self.assertTrue(send_email.called)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_publishers -v`
Expected: import failure

- [ ] **Step 3: Write minimal implementation**

```python
def publish_reports(config: dict, markdown_path: str, html_path: str, summary_text: str) -> dict:
    result = {"discord": {"ok": False}, "email": {"ok": False}}
    discord_cfg = config["notifications"]["discord"]
    email_cfg = config["notifications"]["email"]
    if discord_cfg.get("enabled"):
        try:
            _post_discord(os.environ[discord_cfg["webhook_env"]], summary_text, markdown_path)
            result["discord"]["ok"] = True
        except Exception as exc:
            result["discord"]["error"] = str(exc)
    if email_cfg.get("enabled"):
        try:
            _send_email(email_cfg, html_path)
            result["email"]["ok"] = True
        except Exception as exc:
            result["email"]["error"] = str(exc)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_publishers -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/reporting/publishers.py tests/test_publishers.py
git commit -m "Add Discord and email publishers"
```

### Task 5: Add One-Click Orchestrator and Launcher

**Files:**
- Create: `pipeline/run_push.py`
- Create: `run_daily_push.command`
- Test: `tests/test_run_push.py`
- Modify: `README.md`
- Modify: `README.zh-CN.md`

- [ ] **Step 1: Write the failing orchestrator test**

```python
import unittest
from unittest.mock import patch

from pipeline.run_push import build_delivery_payload


class RunPushTests(unittest.TestCase):
    def test_build_delivery_payload_includes_scan_and_watchlist_symbols(self) -> None:
        scan_output = {
            "summary": {"macro": {"score": 72}, "market_score": 66},
            "watchlist": [{"symbol": "US.NVDA", "final_score": 78, "action": "BUY_CANDIDATE"}],
        }
        watchlist_cfg = {
            "watchlist": [{"symbol": "US.TSLA", "alias": "Tesla", "enabled": True}]
        }
        merged_rows = {
            "US.NVDA": {"community": {"merged_score": 64}, "event": {"merged_score": 58}, "flow": {"merged_score": 60}, "derivatives": {"merged_score": 51}},
            "US.TSLA": {"community": {"merged_score": 62}, "event": {"merged_score": 57}, "flow": {"merged_score": 59}, "derivatives": {"merged_score": 50}},
        }

        payload = build_delivery_payload(scan_output, watchlist_cfg, merged_rows)

        self.assertEqual(payload["market_summary"]["macro_score"], 72)
        self.assertEqual(len(payload["symbols"]), 2)
        self.assertEqual(payload["symbols"][1]["display_name"], "Tesla")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_run_push -v`
Expected: import failure

- [ ] **Step 3: Write minimal implementation**

```python
def build_delivery_payload(scan_output: dict, watchlist_cfg: dict, merged_rows: dict[str, dict]) -> dict:
    tracked = build_tracked_symbols(scan_output.get("watchlist", []), watchlist_cfg)
    symbols = []
    for item in tracked:
        merged = merged_rows.get(item["symbol"], {})
        symbols.append(
            {
                "symbol": item["symbol"],
                "display_name": item.get("alias", item["symbol"]),
                "final_score": next((row["final_score"] for row in scan_output.get("watchlist", []) if row["symbol"] == item["symbol"]), 50),
                "community_score": merged.get("community", {}).get("merged_score", 50),
                "flow_score": merged.get("flow", {}).get("merged_score", 50),
                "event_score": merged.get("event", {}).get("merged_score", 50),
                "derivatives_score": merged.get("derivatives", {}).get("merged_score", 50),
                "technical_score": next((row.get("technical_score", row.get("trigger_score", 50)) for row in scan_output.get("watchlist", []) if row["symbol"] == item["symbol"]), 50),
            }
        )
    return {
        "market_summary": {
            "macro_score": scan_output["summary"]["macro"]["score"],
            "market_score": scan_output["summary"]["market_score"],
            "final_label": "RISK_ON" if scan_output["summary"]["macro"]["score"] >= 60 else "NEUTRAL",
        },
        "symbols": symbols,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_run_push -v`
Expected: `OK`

- [ ] **Step 5: Update docs and launcher**

Add:

```bash
#!/bin/zsh
cd "/Users/xiao1/Documents/Xiao1 trading" || exit 1
python3 -m pipeline.run_push
```

Document in both README files:

- how to edit `config/my_watchlist.yaml`
- how to run `python -m pipeline.run_push`
- how to use `run_daily_push.command`
- which environment variables are required for Discord and Email

- [ ] **Step 6: Run full verification**

Run: `python3 -m unittest discover -s tests -v`
Expected: `OK`

Run: `python3 -m compileall pipeline src research tests`
Expected: exit `0`

- [ ] **Step 7: Commit**

```bash
git add pipeline/run_push.py run_daily_push.command README.md README.zh-CN.md tests/test_run_push.py
git commit -m "Add one-click watchlist push workflow"
```
