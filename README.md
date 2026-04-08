# US Swing Trading System v1

小资金美股 Swing 决策系统 —— 从数据到 watchlist 的完整 pipeline。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置环境变量（FRED key 可选，没有则跳过宏观层）
export FRED_API_KEY=your_key        # https://fred.stlouisfed.org/docs/api/api_key.html
export FINNHUB_API_KEY=your_key     # 可选，用于公司新闻

# 3. 每日收盘后运行
python -m pipeline.run_daily

# 4. 查看结果
cat "/Users/xiao1/Desktop/2026 Day Scan output/daily_scan_latest.md"
```

## 系统架构

```
L1 Event Risk  →  L2 Regime  →  L3 Breadth  →
L4 Sentiment   →  L5 Structure  →  L6 Trigger  →
综合评分 → watchlist (BUY_CANDIDATE / WATCH / PASS / EXIT)
```

| 层级 | 模块 | 一票否决 | 数据来源 |
|------|------|---------|---------|
| L1 Event Risk | event_risk.py | 是 | events_calendar.csv（手工维护） |
| L2 Regime | regime.py | 是 | yfinance (SPY/QQQ/VIX) + FRED（可选） |
| L3 Breadth | breadth.py | 否 | yfinance（宇宙内部计算） |
| L4 Sentiment | sentiment.py | 否 | RSS + CNN F&G |
| L5 Structure | structure.py | 否（筛选） | yfinance |
| L6 Trigger | triggers.py | 否 | yfinance |

## 目录说明

```
config/
  thresholds.yaml      所有可调参数（不改代码只改这里）
  universe.yaml        目标股票宇宙 + 板块映射
  events_calendar.csv  高影响事件日历（每月手工更新）

src/adapters/          数据适配器层
src/features/          L1-L6 因子计算

pipeline/
  run_daily.py         每日主流水线（每天跑一次）

research/
  factor_tests.py      单因子测试 + Ablation 框架

outputs/
  watchlists/          每日 JSON 输出
  reports/             每日 Markdown 报告
  trade_journal.csv    交易记录（手工填写）
```

## 单文件输出

默认情况下，本地运行会把完整扫描结果写到：

```bash
/Users/xiao1/Desktop/2026 Day Scan output/daily_scan_latest.md
```

如果你想换目录，可以在运行前设置：

```bash
export DAILY_SCAN_OUTPUT_DIR=/your/output/folder
python -m pipeline.run_daily
```

## 常用命令

```bash
# 指定日期运行
python -m pipeline.run_daily --date 2026-04-02

# 强制刷新价格缓存
python -m pipeline.run_daily --refresh-prices

# 单因子测试
python -m research.factor_tests --factor rs_vs_spy --forward 10

# 全因子测试 + Ablation
python -m research.factor_tests --all --forward 10
```

## 与 Futu 指标对接

Python pipeline 负责 L1-L4（宏观/环境/情绪），
Futu 指标负责 L5-L6（个股结构/触发），两者互补不重复：

```
pipeline 每日输出 outputs/watchlists/latest.json
  → 手动把 regime color + watchlist 对照 Futu 图表
  → Futu 图表确认 GSQUEEZE 状态 + BIGMONEY 信号
  → 综合判断入场
```

## 风控规则（强制，不可例外）

- 单笔期权亏损 > 50%：立即平仓
- 单标的持仓 > 总资金 40%：禁止加仓
- Regime Red：停止新多头开仓
- 时间止损：入场后 7 日无启动 → 减半或退出
- 宏观评分 40-60 中性区：不入场

## v1.1 预留接口

- `sentiment.py` 中 `_rule_score()` 可替换为 FinBERT 模型
- `factor_tests.py` 输出的 CSV 可接 sklearn 做简单的因子组合权重优化
- `events_calendar.csv` 可接 SEC EDGAR API 自动更新财报日期

## GitHub Actions 自动运行

仓库已包含工作流文件：

```text
.github/workflows/daily-scan.yml
```

默认行为：

- 工作日 `20:30 UTC` 自动运行一次
- GitHub 页面支持手动点击 `Run workflow`
- 自动上传 `daily_scan_latest.md` 作为 artifact
- 如果配置了 secrets，会自动推送到 Telegram 和 Discord

### 需要配置的 GitHub Secrets

- `FRED_API_KEY`：可选
- `FINNHUB_API_KEY`：可选
- `TELEGRAM_BOT_TOKEN`：Telegram BotFather 创建的 bot token
- `TELEGRAM_CHAT_ID`：接收报告的 chat id
- `DISCORD_WEBHOOK_URL`：Discord channel webhook

### Telegram / Discord 说明

- Telegram 会收到完整 Markdown 报告附件
- Discord 会收到同一份 Markdown 报告附件
- 如果对应 secret 没填，工作流会跳过该推送步骤，不会让扫描失败
