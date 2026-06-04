"""
pipeline/run_daily.py
─────────────────────────────────────────────────────────────
每日收盘后主流水线（每天跑一次）

执行顺序：
  L1 Event Risk  →  L2 Macro  →  L3 Regime/Breadth  →
  L4 Community   →  L5 Flow/Event  →  L6 Structure/Trigger  →
  综合评分  →  输出 watchlist

输出文件：
  默认：
    /Users/xiao1/Desktop/2026 Day Scan output/daily_scan_latest.md
  可通过环境变量 DAILY_SCAN_OUTPUT_DIR 覆盖输出目录

用法：
  python -m pipeline.run_daily
  python -m pipeline.run_daily --date 2026-04-02
  python -m pipeline.run_daily --refresh-prices
"""

from __future__ import annotations
import argparse
from collections import Counter
import json
import logging
import os
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

# ── 路径修复（让相对导入正常工作）─────────────────────────────
# pipeline/run_daily.py → parents[0]=pipeline/ → parents[1]=trading_system/ (root)
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapters.price_client import PriceClient
from adapters.fred_client  import FredClient
from adapters.news_client  import NewsClient
from adapters.community_client import CommunitySnapshotClient
from adapters.flow_client import FlowSnapshotClient
from features.event_risk   import check_event_risk
from features.macro        import compute_macro_context
from features.regime       import compute_regime
from features.breadth      import compute_breadth
from features.sentiment    import compute_sentiment
from features.community    import compute_community_sentiment
from features.flow         import compute_flow_signal
from features.stock_events import compute_stock_event_signal
from features.scoring      import compute_composite_score
from features.structure    import screen_structure
from features.triggers     import screen_triggers

# ── 日志 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── 输出目录 ──────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = Path("/Users/xiao1/Desktop/2026 Day Scan output")
OUTPUT_DIR = Path(os.environ.get("DAILY_SCAN_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LATEST_REPORT_PATH = OUTPUT_DIR / "daily_scan_latest.md"


# ═════════════════════════════════════════════════════════════
# 辅助函数
# ═════════════════════════════════════════════════════════════
def load_config() -> tuple[dict, dict, list[str]]:
    """加载 thresholds / universe 配置"""
    cfg_dir = ROOT / "config"
    with open(cfg_dir / "thresholds.yaml") as f:
        thresholds = yaml.safe_load(f)
    with open(cfg_dir / "universe.yaml") as f:
        universe_cfg = yaml.safe_load(f)
    stocks = universe_cfg.get("stocks", [])
    return thresholds, universe_cfg, stocks


def decide_action(
    event_gate:  bool,
    macro_gate:  bool,
    regime_gate: bool,
    trigger_signal: str,
    final_label: str,
) -> str:
    """
    最终操作决策。
    优先尊重大盘/宏观，再用技术信号决定执行节奏。
    返回：PASS / WATCH / BUY_CANDIDATE / REDUCE / EXIT
    """
    if not event_gate:
        return "PASS"  # 事件窗口封锁
    if not macro_gate:
        return "PASS"  # 市场环境 veto
    if not regime_gate:
        return "REDUCE"
    if trigger_signal == "EXIT":
        return "EXIT"
    if trigger_signal == "BUY_CANDIDATE" and final_label == "BUY_CANDIDATE":
        return "BUY_CANDIDATE"
    if trigger_signal in ("BUY_CANDIDATE", "WATCH") and final_label in ("BUY_CANDIDATE", "WATCH"):
        return "WATCH"
    return "PASS"


# ═════════════════════════════════════════════════════════════
# 主流水线
# ═════════════════════════════════════════════════════════════
def run_pipeline(
    run_date:      date | None = None,
    refresh_prices: bool = False,
    fred_key:       str | None = None,
    finnhub_key:    str | None = None,
) -> dict:
    run_date = run_date or date.today()
    logger.info("═══ Daily Pipeline: %s ═══", run_date)

    thresholds, universe_cfg, stocks = load_config()
    all_etfs = list(
        set(
            universe_cfg.get("regime_etfs", [])
            + list(universe_cfg.get("sector_etfs", {}).keys())
        )
    )
    stock_sector_map = universe_cfg.get("stock_sector_map", {})
    price_client = PriceClient(cache=True)

    logger.info("L1 Event Risk...")
    event_result = check_event_risk(
        block_window_hours=thresholds.get("event_risk", {}).get("block_window_hours", 24),
        warn_window_hours=thresholds.get("event_risk", {}).get("warn_window_hours", 48),
    )
    logger.info("  → %s  %s", event_result.level, event_result.note)

    logger.info("Fetching ETF prices (%d symbols)...", len(all_etfs))
    etf_prices = price_client.fetch_bulk(
        all_etfs,
        period="1y",
        batch_size=20,
        sleep=0.5,
        force_refresh=refresh_prices,
    )

    macro_df = None
    if fred_key:
        try:
            logger.info("Fetching FRED macro data...")
            macro_df = FredClient(api_key=fred_key).fetch_series(lookback_days=60)
        except Exception as e:
            logger.warning("FRED failed (continuing without macro): %s", e)
    else:
        logger.info("No FRED_API_KEY — skipping macro overlay")

    logger.info("L2 Macro Context...")
    macro = compute_macro_context(
        macro_df=macro_df,
        event_risk=event_result,
        cfg=thresholds.get("macro", {}),
    )
    logger.info("  → %s (score=%.0f)  %s", macro.label, macro.score, macro.note)

    logger.info("L3 Market Regime...")
    regime = compute_regime(etf_prices, macro_df, cfg=thresholds.get("regime", {}))
    logger.info("  → %s (score=%.0f)  %s", regime.color, regime.score, regime.note)

    logger.info("Fetching universe prices (%d stocks)...", len(stocks))
    universe_prices = price_client.fetch_bulk(
        stocks,
        period="2y",
        batch_size=40,
        sleep=1.0,
        force_refresh=refresh_prices,
    )
    logger.info("  → %d symbols loaded", len(universe_prices))

    logger.info("L4 Breadth & Leadership...")
    breadth = compute_breadth(
        universe_prices=universe_prices,
        etf_prices=etf_prices,
        sector_etfs=list(universe_cfg.get("sector_etfs", {}).keys()),
        cfg=thresholds.get("breadth", {}),
    )
    logger.info("  → %s (score=%.0f)  %s", breadth.level, breadth.score, breadth.note)

    logger.info("Legacy market sentiment...")
    headlines = pd.DataFrame()
    news_client = NewsClient(finnhub_key=finnhub_key)
    try:
        headlines = news_client.fetch_rss_headlines(hours_back=24)
        logger.info("  → %d headlines fetched", len(headlines))
    except Exception as e:
        logger.warning("News fetch failed: %s", e)

    legacy_sentiment = compute_sentiment(
        headlines=headlines,
        vix_level=regime.vix_level,
        vix_5d_chg=regime.vix_5d_chg,
        fetch_fg=True,
    )
    logger.info("  → combined adj: %+.1f  %s", legacy_sentiment.combined, legacy_sentiment.note)

    community_posts = CommunitySnapshotClient.from_config(thresholds).fetch()
    community_market = compute_community_sentiment(
        community_posts,
        cfg=thresholds.get("community", {}),
    )
    flow_snapshots = FlowSnapshotClient.from_config(thresholds).fetch()
    logger.info(
        "Community rows=%d  Flow rows=%d",
        len(community_posts),
        len(flow_snapshots),
    )

    logger.info("L6 Structure Screening (%d symbols)...", len(universe_prices))
    structure_results = screen_structure(
        universe_prices=universe_prices,
        etf_prices=etf_prices,
        stock_sector_map=stock_sector_map,
        cfg=thresholds.get("structure", {}),
    )
    candidates = [r for r in structure_results if r.structure_ok]
    logger.info("  → %d / %d pass structure", len(candidates), len(structure_results))

    logger.info("L7 Trigger Scanning (%d candidates)...", len(candidates))
    trigger_results = screen_triggers(
        candidates=candidates[:50],
        universe_prices=universe_prices,
        cfg=thresholds.get("trigger", {}),
    )

    market_score = round(regime.score * 0.55 + breadth.score * 0.45, 2)
    company_news_by_symbol: dict[str, pd.DataFrame] = {}
    if news_client.finnhub_key:
        for symbol in [result.symbol for result in trigger_results[:25]]:
            company_news = news_client.fetch_company_news(symbol, days=3)
            if not company_news.empty:
                company_news = company_news.copy()
                company_news["symbol"] = symbol
            company_news_by_symbol[symbol] = company_news
    else:
        logger.info("No FINNHUB_API_KEY — skipping per-symbol company news events")

    watchlist = []
    for tr in trigger_results:
        st = next((r for r in candidates if r.symbol == tr.symbol), None)
        technical_score = round(((st.score if st else 0) * 0.45) + tr.score * 0.55, 2)
        community_signal = compute_community_sentiment(
            community_posts,
            symbol=tr.symbol,
            cfg=thresholds.get("community", {}),
        )
        flow_signal = compute_flow_signal(
            flow_snapshots,
            symbol=tr.symbol,
            cfg=thresholds.get("flow", {}),
        )
        stock_event_signal = compute_stock_event_signal(
            company_news_by_symbol.get(tr.symbol, pd.DataFrame()),
            symbol=tr.symbol,
            cfg=thresholds.get("stock_events", {}),
        )
        composite = compute_composite_score(
            macro_score=macro.score,
            market_score=market_score,
            community_score=community_signal.overall_score,
            flow_score=flow_signal.score,
            event_score=stock_event_signal.score,
            technical_score=technical_score,
            cfg=thresholds.get("scoring", {}),
        )
        action = decide_action(
            event_gate=event_result.gate_pass,
            macro_gate=macro.gate_pass,
            regime_gate=regime.gate_pass,
            trigger_signal=tr.signal,
            final_label=composite.label,
        )
        watchlist.append(
            {
                "symbol": tr.symbol,
                "action": action,
                "final_score": composite.score,
                "close": st.close if st else 0,
                "rs_vs_spy": round((st.rs_vs_spy if st else 0) * 100, 2),
                "trigger": tr.signal,
                "trigger_score": tr.score,
                "macro_score": macro.score,
                "market_score": market_score,
                "community_score": community_signal.overall_score,
                "community_label": community_signal.symbol_label,
                "flow_score": flow_signal.score,
                "flow_label": flow_signal.label,
                "event_score": stock_event_signal.score,
                "event_label": stock_event_signal.label,
                "technical_score": technical_score,
                "stop_below": tr.stop_below,
                "entry_zone": f"{tr.entry_zone_low:.2f}-{tr.entry_zone_high:.2f}",
                "atr14": st.atr14 if st else 0,
                "note": (
                    f"Macro {macro.label}/{macro.score:.0f} | "
                    f"Community {community_signal.symbol_label}/{community_signal.overall_score:.0f} | "
                    f"Flow {flow_signal.label}/{flow_signal.score:.0f} | "
                    f"Event {stock_event_signal.label}/{stock_event_signal.score:.0f} | "
                    f"Structure: {st.note if st else 'N/A'} | Trigger: {tr.note}"
                ),
            }
        )

    priority = {"BUY_CANDIDATE": 0, "WATCH": 1, "PASS": 2, "EXIT": 3, "REDUCE": 4}
    watchlist.sort(key=lambda item: (priority.get(item["action"], 5), -item["final_score"]))
    action_counts = Counter(w["action"] for w in watchlist)

    output = {
        "run_date": run_date.isoformat(),
        "generated": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "event_risk": event_result.to_dict(),
            "macro": macro.to_dict(),
            "regime": regime.to_dict(),
            "breadth": breadth.to_dict(),
            "legacy_sentiment": legacy_sentiment.to_dict(),
            "community_market": community_market.to_dict(),
            "market_score": market_score,
            "community_rows": len(community_posts),
            "flow_rows": len(flow_snapshots),
            "universe_count": len(universe_prices),
            "structure_pass": len(candidates),
            "trigger_scanned": len(trigger_results),
            "buy_candidates": sum(1 for w in watchlist if w["action"] == "BUY_CANDIDATE"),
            "watch_count": sum(1 for w in watchlist if w["action"] == "WATCH"),
            "action_counts": dict(action_counts),
        },
        "watchlist": watchlist,
    }

    _write_report(
        output=output,
        structure_results=structure_results,
        trigger_results=trigger_results,
        path=LATEST_REPORT_PATH,
    )

    logger.info(
        "═══ Done — %d buy candidates, %d watch ═══",
        output["summary"]["buy_candidates"],
        output["summary"]["watch_count"],
    )
    return output


def _top_structure_rows(results: list, limit: int = 15) -> list[str]:
    rows = [
        "| Symbol | Structure | Score | Close | RS vs SPY | RS vs Sector | ATR14 | Note |",
        "|--------|-----------|-------|-------|-----------|--------------|-------|------|",
    ]
    for r in results[:limit]:
        rows.append(
            f"| {r.symbol} | {'PASS' if r.structure_ok else 'FAIL'} | {r.score:.1f} | "
            f"{r.close:.2f} | {r.rs_vs_spy*100:+.2f}% | {r.rs_vs_sector*100:+.2f}% | "
            f"{r.atr14:.2f} | {r.note} |"
        )
    return rows


def _structure_fail_reason_lines(results: list, limit: int = 8) -> list[str]:
    fail_counter: Counter[str] = Counter()
    for r in results:
        if r.structure_ok or r.note == "OK":
            continue
        for reason in [p.strip() for p in r.note.split(",") if p.strip()]:
            fail_counter[reason] += 1

    if not fail_counter:
        return ["- 没有结构层失败项，全部通过。"]

    return [
        f"- `{reason}`: {count} symbols"
        for reason, count in fail_counter.most_common(limit)
    ]


def _trigger_rows(results: list) -> list[str]:
    rows = [
        "| Symbol | Signal | Score | Near EMA | Volume Contracted | Restart Volume | Stop | Entry Zone | Note |",
        "|--------|--------|-------|----------|-------------------|----------------|------|------------|------|",
    ]
    for r in results:
        near_label = "21EMA" if r.pullback_near_21 else ("50EMA" if r.pullback_near_50 else "No")
        rows.append(
            f"| {r.symbol} | {r.signal} | {r.score:.1f} | {near_label} | "
            f"{r.volume_contracted} | {r.restart_volume} | {r.stop_below:.2f} | "
            f"{r.entry_zone_low:.2f}-{r.entry_zone_high:.2f} | {r.note} |"
        )
    return rows


def _final_watchlist_rows(watchlist: list[dict]) -> list[str]:
    rows = [
        "| Symbol | Action | Final | Macro | Market | Community | Flow | Event | Tech | Close | Trigger | Stop | Entry Zone | Note |",
        "|--------|--------|-------|-------|--------|-----------|------|-------|------|-------|---------|------|------------|------|",
    ]
    for w in watchlist:
        rows.append(
            f"| {w['symbol']} | {w['action']} | {w['final_score']:.1f} | {w['macro_score']:.1f} | "
            f"{w['market_score']:.1f} | {w['community_label']}/{w['community_score']:.1f} | "
            f"{w['flow_label']}/{w['flow_score']:.1f} | {w['event_label']}/{w['event_score']:.1f} | "
            f"{w['technical_score']:.1f} | {w['close']:.2f} | {w['trigger']} | "
            f"{w['stop_below']:.2f} | {w['entry_zone']} | {w['note']} |"
        )
    return rows


def _write_report(
    output: dict,
    structure_results: list,
    trigger_results: list,
    path: Path,
) -> None:
    s = output["summary"]
    macro = s["macro"]
    r = s["regime"]
    b = s["breadth"]
    ev = s["event_risk"]
    legacy_sent = s["legacy_sentiment"]
    community = s["community_market"]
    action_counts = s.get("action_counts", {})

    lines = [
        f"# Daily Scan Report  {output['run_date']}",
        "",
        "## 输出说明",
        "",
        f"- 本文件是本次扫描的唯一输出，路径：`{path}`",
        "- 数据来自 yfinance / RSS / CNN Fear & Greed / FRED（如果配置了 key）",
        "- 如提供了 Futu / moomoo 导出快照，会自动纳入社区情绪与资金流打分",
        "- 价格是日线级扫描结果，更适合收盘后复盘，不是券商级盘中逐笔实时信号",
        "",
        "## 决策漏斗",
        "",
        f"1. 股票宇宙：`{s['universe_count']}` symbols",
        f"2. L1 Event Risk：`{ev['level']}` -> {ev['note']}",
        f"3. L2 Macro：`{macro['label']}` (score={macro['score']}) -> {macro['note']}",
        f"4. L3 Market：Regime `{r['color']}` + Breadth `{b['level']}` -> blended `{s['market_score']:.1f}`",
        f"5. L4 Community：market `{community['market_label']}` / score `{community['market_score']}`",
        f"6. L5 Flow / Event：community rows `{s['community_rows']}` | flow rows `{s['flow_rows']}`",
        f"7. L6 Structure：`{s['structure_pass']}` / `{s['universe_count']}` pass",
        f"8. L7 Trigger：扫描 `Top {s['trigger_scanned']}` structure candidates",
        f"9. 最终结果：BUY `{s['buy_candidates']}` | WATCH `{s['watch_count']}` | "
        f"其他 `{sum(v for k, v in action_counts.items() if k not in ('BUY_CANDIDATE', 'WATCH'))}`",
        "",
        "## 环境判断细节",
        "",
        "| 层级 | 状态 | 关键数据 | 注释 |",
        "|------|------|----------|------|",
        f"| L1 Event Risk | **{ev['level']}** | 24h={ev['events_24h_count']} / 48h={ev['events_48h_count']} | {ev['note']} |",
        f"| L2 Macro | **{macro['label']}** | score={macro['score']}, gate={macro['gate_pass']} | {macro['note']} |",
        f"| L3 Regime | **{r['color']}** | VIX={r['vix_level']} ({r['vix_state']}), 5d={r['vix_5d_chg']*100:+.1f}% | {r['note']} |",
        f"| L4 Breadth | **{b['level']}** | >200EMA={b['pct_above_200ema']}%, NH/NL={b['nh_nl_ratio']} | {b['note']} |",
        f"| L5 Community | **{community['market_label']}** | score={community['market_score']}, rows={s['community_rows']} | {community['note']} |",
        f"| Legacy Sentiment | **{legacy_sent['sentiment_adjustment']:+.2f}** | Market={legacy_sent['market_sentiment_score']:+.2f}, VIX={legacy_sent['vix_sentiment_score']:+.2f}, F&G={legacy_sent['fear_greed_index']:.1f} | {legacy_sent['note']} |",
        "",
        f"- SPY > 200EMA: `{r['spy_above_200']}`",
        f"- QQQ 21EMA > 50EMA: `{r['qqq_21_gt_50']}`",
        f"- DXY 5d change: `{r['dxy_5d_pct']:+.2f}%`",
        f"- US10Y 5d move: `{r['us10y_5d_bps']:+.2f} bps`",
        f"- Leading sectors: `{', '.join(b['leading_sectors']) if b['leading_sectors'] else 'None'}`",
        f"- Lagging sectors: `{', '.join(b['lagging_sectors']) if b['lagging_sectors'] else 'None'}`",
        "",
        "## L6 Structure 明细",
        "",
    ]

    if structure_results:
        lines.extend(_top_structure_rows(structure_results))
    else:
        lines.append("- 本次没有可用的结构层结果。")

    lines += [
        "",
        "### Structure 淘汰原因统计",
        "",
    ]

    lines.extend(_structure_fail_reason_lines(structure_results))

    lines += [
        "",
        "## L7 Trigger 扫描结果",
        "",
    ]

    if trigger_results:
        lines.extend(_trigger_rows(trigger_results))
    else:
        lines.append("- 本次没有进入 Trigger 扫描的候选。")

    lines += [
        "",
        "## 最终动作清单",
        "",
    ]

    if output["watchlist"]:
        lines.extend(_final_watchlist_rows(output["watchlist"]))
    else:
        lines.append("- 本次没有生成最终 watchlist。")

    lines += [
        "",
        "## 使用说明",
        "",
        "- `BUY_CANDIDATE`：环境允许，结构和触发都相对完整，可以进入人工复核。",
        "- `WATCH`：结构或回踩质量不错，但还缺少再启动确认。",
        "- `PASS`：不满足开仓条件，今天不处理。",
        "- `EXIT`：若已有仓位，当前价格/量能结构提示应偏保守。",
        "- `REDUCE`：通常出现在环境层不够理想时，即使个股形态还在，也建议降风险。",
        "",
        "## Structured Snapshot",
        "",
        "```json",
        json.dumps(output, ensure_ascii=False, indent=2),
        "```",
        "",
        f"*Generated: {output['generated']}*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written: %s", path)


# ═════════════════════════════════════════════════════════════
# CLI 入口
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Daily trading pipeline")
    parser.add_argument("--date",           type=str,  help="Run date YYYY-MM-DD")
    parser.add_argument("--refresh-prices", action="store_true",
                        help="Force re-download all prices")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()

    try:
        result = run_pipeline(
            run_date=run_date,
            refresh_prices=args.refresh_prices,
            fred_key=os.environ.get("FRED_API_KEY"),
            finnhub_key=os.environ.get("FINNHUB_API_KEY"),
        )
        # 打印摘要到终端
        print("\n" + "═"*55)
        print(f"  MARKET SUMMARY  {run_date}")
        print("═"*55)
        r = result["summary"]["regime"]
        b = result["summary"]["breadth"]
        print(f"  Regime:  {r['color']}  (VIX {r['vix_level']:.1f})")
        print(f"  Breadth: {b['level']}  (score {b['score']:.0f})")
        print(f"  Buy Candidates: {result['summary']['buy_candidates']}")
        print(f"  Watch:          {result['summary']['watch_count']}")
        print(f"  Report:         {LATEST_REPORT_PATH}")
        print("─"*55)
        for w in result["watchlist"][:10]:
            if w["action"] in ("BUY_CANDIDATE","WATCH"):
                print(f"  {w['symbol']:<6} {w['action']:<14} "
                      f"score={w['final_score']:5.1f}  "
                      f"RS={w['rs_vs_spy']:+.1f}%  "
                      f"stop=${w['stop_below']:.2f}")
        print("═"*55 + "\n")

    except Exception:
        logger.error("Pipeline failed:\n%s", traceback.format_exc())
        sys.exit(1)
