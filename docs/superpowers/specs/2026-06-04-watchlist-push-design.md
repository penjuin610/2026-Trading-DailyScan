# Watchlist Push Design

## Goal

Build a first-version `one-click` local workflow for the trading repository that:

- runs without OpenD
- aggregates both `Futu` and `moomoo` content-side skills together in all language contexts
- combines `community + event + flow + derivatives` with the existing `macro + market + light technical` engine
- includes both scanned candidates and a user-maintained watchlist
- generates clear visual score output and ratio breakdowns
- sends results to `Discord + Email`

This version should prioritize reliability and easy daily use over UI complexity.

## User Intent

The target usage is:

1. user clicks a desktop launcher or runs one command
2. system evaluates market context and tracked stocks
3. system generates a readable report with visualization and score breakdowns
4. system pushes the result to Discord and Email
5. report includes both system scan output and manually maintained watchlist names such as `TSLA`, `PDD`, and `FIG`

## Scope

### In Scope

- local single-command execution
- optional desktop double-click launcher
- watchlist config in repository
- dual-platform aggregation from `Futu + moomoo`
- external signal ingestion for:
  - community sentiment
  - company/news event interpretation
  - capital anomaly / flow
  - derivatives anomaly
- market context reuse from existing local pipeline:
  - event risk
  - macro
  - regime
  - breadth
- Markdown + HTML + JSON output
- push to Discord webhook and Email SMTP
- graceful degradation when part of the external data is missing

### Out of Scope

- OpenD integration
- brokerage account reads
- live trading
- website UI
- persistent database
- WhatsApp first-version delivery

## Architecture

### Execution Shape

The first version will use:

- `python -m pipeline.run_push`
- optional `run_daily_push.command`

The existing [pipeline/run_daily.py](/Users/xiao1/Documents/Xiao1 trading/pipeline/run_daily.py:1) remains the core local scoring pipeline. A new top-level orchestrator wraps it and adds:

- watchlist expansion
- dual-platform external signal collection
- report rendering
- notification publishing

### Data Flow

1. load repository config
2. load manual watchlist
3. run local market pipeline
4. build symbol universe:
   - scanned candidates
   - enabled watchlist symbols
5. collect dual-platform external signals
6. normalize and merge external signals into per-symbol snapshots
7. calculate final watchlist-facing summary records
8. render Markdown report
9. render HTML report
10. write JSON snapshot
11. publish to Discord and Email

## Layer-to-Skill Mapping

### Local Factors

- `Event Risk` -> local `event_risk.py`
- `Macro` -> local `macro.py`
- `Regime` -> local `regime.py`
- `Breadth` -> local `breadth.py`
- `Technical` -> local `structure.py` + `triggers.py`

### External Factors

- `Community`
  - `futu-comment-sentiment`
  - `moomoo-comment-sentiment`
- `Event / News`
  - `futu-stock-digest`
  - `moomoo-stock-digest`
  - optional detail expansion:
    - `futu-news-search`
    - `moomoo-news-search`
- `Flow`
  - `futu-capital-anomaly`
  - `moomoo-capital-anomaly`
- `Derivatives`
  - `futu-derivatives-anomaly`
  - `moomoo-derivatives-anomaly`

## Signal Combination Rules

### Platform Policy

For both Chinese and English contexts:

- always run `Futu + moomoo` together
- do not route by language to a single platform
- keep separate platform results and also compute a merged interpretation

### Community Merge

Per symbol, preserve:

- Futu bullish / bearish / neutral ratio
- moomoo bullish / bearish / neutral ratio
- merged bullish / bearish / neutral ratio
- merged sentiment score

### Event Merge

Per symbol, preserve:

- Futu digest direction
- moomoo digest direction
- merged event stance
- top evidence headlines / links

### Flow Merge

Per symbol, preserve:

- Futu capital anomaly result
- moomoo capital anomaly result
- merged flow stance
- anomaly presence summary

### Derivatives Merge

Per symbol, preserve:

- Futu derivatives anomaly result
- moomoo derivatives anomaly result
- merged derivatives stance

### Final Symbol Output

Each symbol record should expose:

- `macro_score`
- `market_score`
- `community_score`
- `flow_score`
- `event_score`
- `derivatives_score`
- `technical_score`
- `final_score`
- `final_label`
- `summary_line`

The model remains macro-led. Technical stays a light confirmation layer.

## Weights

First-version proposed weights:

- Macro: 22%
- Market: 18%
- Community: 18%
- Flow: 14%
- Event: 14%
- Derivatives: 9%
- Technical: 5%

Rationale:

- macro + market remain the top gate
- community / flow / event form the main conviction layer
- derivatives matter for options-oriented swing decisions
- technical only confirms execution quality

## Watchlist Config

Add a new file:

- [config/my_watchlist.yaml](/Users/xiao1/Documents/Xiao1 trading/config/my_watchlist.yaml:1)

Planned structure:

```yaml
watchlist:
  - symbol: US.TSLA
    alias: Tesla
    enabled: true
  - symbol: US.PDD
    alias: PDD
    enabled: true
  - symbol: US.FIG
    alias: Figma
    enabled: true

notifications:
  discord:
    enabled: true
    webhook_env: DISCORD_WEBHOOK_URL
  email:
    enabled: true
    smtp_host_env: SMTP_HOST
    smtp_port_env: SMTP_PORT
    username_env: SMTP_USERNAME
    password_env: SMTP_PASSWORD
    from_env: SMTP_FROM
    to_env: SMTP_TO

report:
  include_scan_candidates: true
  include_watchlist: true
  top_scan_limit: 15
  chart_style: bar_and_table
```

### User Editing Model

The user should be able to:

- add a stock by adding one item
- disable a stock by setting `enabled: false`
- rename display label by editing `alias`

README docs will explain this directly with examples.

## Output Files

The first version will generate:

- `outputs/reports/daily_push_latest.md`
- `outputs/reports/daily_push_latest.html`
- `outputs/data/latest_snapshot.json`

## Report Design

### Markdown Report

Primary use:

- Discord attachment
- terminal-friendly review

Sections:

1. market summary
2. top scan candidates
3. manual watchlist status
4. per-symbol factor table
5. evidence appendix

### HTML Report

Primary use:

- Email body or attachment
- local visual inspection

Planned visual components:

- horizontal score bars
- bullish / neutral / bearish ratio bars
- platform comparison table
- summary heat table

## Visualization Requirements

Per symbol visualization should include:

- factor score bar set:
  - Macro
  - Market
  - Community
  - Flow
  - Event
  - Derivatives
  - Technical
  - Final
- community ratio view:
  - Futu
  - moomoo
  - merged
- concise summary label

This should make both the raw ratios and the final score visible at a glance.

## Notification Design

### Discord

Send:

- short summary message in webhook content
- attach Markdown report

Summary should include:

- overall market stance
- top scan candidates
- watchlist highlights

### Email

Send:

- HTML summary in email body
- optional Markdown or HTML attachment

Email should remain readable on mobile.

## Failure / Degradation Policy

### External Platform Failure

If one platform fails:

- continue with the other platform
- mark missing side as unavailable
- still compute merged output from available side

If both sides fail for a factor:

- mark factor as `N/A`
- reduce confidence note
- continue overall run

### Notification Failure

If Discord fails:

- still send Email

If Email fails:

- still send Discord

If both fail:

- still write local report artifacts

### Sparse Symbol Data

If a watchlist symbol has insufficient external signal:

- still include the symbol
- mark it as `data sparse`
- keep visible local market context

## Files To Add or Change

### New Files

- `config/my_watchlist.yaml`
- `pipeline/run_push.py`
- `run_daily_push.command`
- `src/adapters/external_sentiment_client.py`
- `src/adapters/external_event_client.py`
- `src/adapters/external_anomaly_client.py`
- `src/reporting/render_markdown.py`
- `src/reporting/render_html.py`
- `src/reporting/publishers.py`
- tests for config loading, merge logic, renderers, and publisher behavior

### Existing Files To Update

- `README.md`
- `README.zh-CN.md`
- `config/thresholds.yaml`
- possibly `pipeline/run_daily.py` only if shared utilities should be extracted

## Testing Strategy

Use test-first for:

- watchlist config loading
- symbol set merge between scan results and watchlist
- dual-platform sentiment merge
- dual-platform event merge
- dual-platform anomaly merge
- report rendering sections
- notification publisher fallback behavior

Verification commands at minimum:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall pipeline src research tests
```

## GitHub Sync Plan

After implementation is approved and completed:

1. commit the implementation changes
2. push to the existing GitHub repository
3. keep `README.md` as English primary
4. keep `README.zh-CN.md` as Chinese counterpart

## Risks

- external skill/script availability may vary by environment
- HTML rendering can become overcomplicated if visualization scope grows
- Email SMTP config is often the most fragile part of first-run setup
- platform data may be sparse for some stocks

## Recommendation

Implement in two layers:

1. `run_push.py` + watchlist config + Markdown/HTML + Discord/Email
2. stronger renderer polish and richer anomaly details

This keeps first delivery practical, fast, and aligned with daily use.
