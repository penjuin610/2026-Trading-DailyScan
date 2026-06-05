# 美股 Swing 交易系统 v2

[English Version](README.md)

这是一个面向 `正股 + 期权 swing` 的美股交易工作流，核心思想是 `宏观和大盘先定方向，再用情绪、资金流和个股事件做增强确认，技术只做轻量执行判断`。

系统优先关注：

- 宏观方向和重大事件风险
- 大盘 Regime 与 Breadth
- Futu / moomoo 生态评论情绪
- 资金流和个股事件
- 轻量技术执行确认

技术结构依然重要，但不再主导总决策。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 可选环境变量
export FRED_API_KEY=your_key
export FINNHUB_API_KEY=your_key

# 3. 可选的 Futu/moomoo 快照输入
# data/inputs/community/futu_comments.json
# data/inputs/community/moomoo_comments.json
# data/inputs/flows/futu_capital.json
# data/inputs/flows/moomoo_capital.json

# 4. 每日收盘后运行
python -m pipeline.run_daily

# 5. 一键运行推送版
python -m pipeline.run_push

# 6. 查看报告
cat "/Users/xiao1/Desktop/2026 Day Scan output/daily_scan_latest.md"
```

## 一键推送入口

你可以用两种方式运行：

```bash
python -m pipeline.run_push
```

或者直接双击：

```bash
./run_daily_push.command
```

它会生成：

- `daily_push_latest.md`
- `daily_push_latest.html`
- `latest_snapshot.json`

并可选推送到 Discord 和 Email。

## 怎么改自己的自选名单

直接编辑 [config/my_watchlist.yaml](/Users/xiao1/Documents/Xiao1 trading/config/my_watchlist.yaml:1)。

示例：

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
    enabled: false
```

你只需要记住三件事：

- 加股票：复制一条，改 `symbol`
- 改显示名：改 `alias`
- 暂时不看：把 `enabled` 改成 `false`

## Discord 和 Email 配置

先在 `config/my_watchlist.yaml` 里把通知开关打开，再设置对应环境变量。

Discord：

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

Email：

```bash
export SMTP_HOST="smtp.example.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your_user"
export SMTP_PASSWORD="your_password"
export SMTP_FROM="from@example.com"
export SMTP_TO="to@example.com"
```

## 系统架构

```text
L1 Event Risk  ->  L2 Macro  ->  L3 Regime/Breadth  ->
L4 Community   ->  L5 Flow/Event  ->  L6 Structure/Trigger  ->
Composite Score -> watchlist
```

| 层级 | 模块 | 一票否决 | 数据来源 |
|------|------|------|------|
| L1 Event Risk | `event_risk.py` | 是 | `events_calendar.csv` |
| L2 Macro | `macro.py` | 是 | FRED + 重大事件窗口 |
| L3 Regime | `regime.py` | 是 | `yfinance` + 可选 FRED |
| L4 Breadth | `breadth.py` | 否 | 股票池内部广度 |
| L5 Community | `community.py` | 否 | Futu/moomoo 评论情绪快照 |
| L6 Flow / Event | `flow.py` + `stock_events.py` | 否 | Futu/moomoo 资金流快照 + Finnhub 公司新闻 |
| L7 Structure / Trigger | `structure.py` + `triggers.py` | 否 | `yfinance` |

## 核心权重

- Macro：25%
- Market：20%
- Community：20%
- Flow：15%
- Stock Events：15%
- Technical：5%

这套权重是典型的 swing 思路：先看 `宏观 + 大盘` 决定应不应该承担多头风险，再看情绪、资金和事件是否支持，最后让技术层负责时机确认。

## 项目结构

```text
config/
  thresholds.yaml
  universe.yaml
  events_calendar.csv

src/adapters/
  price_client.py
  fred_client.py
  news_client.py
  community_client.py
  flow_client.py

src/features/
  event_risk.py
  macro.py
  regime.py
  breadth.py
  community.py
  flow.py
  stock_events.py
  structure.py
  triggers.py
  scoring.py

pipeline/
  run_daily.py

research/
  factor_tests.py

tests/
  P0 逻辑和升级后因子层的回归测试
```

## Futu / moomoo 对接方式

当前版本优先支持“快照驱动接入”。

1. 从 Futu/moomoo 导出或生成评论情绪快照
2. 从 Futu/moomoo 导出或生成资金流快照
3. 放到 `data/inputs/...`
4. 运行 pipeline，系统会自动把这些数据纳入评分

社区情绪快照示例：

```json
[
  {"symbol": "MARKET", "sentiment": "bullish", "mentions": 42},
  {"symbol": "NVDA", "sentiment": "bearish", "mentions": 8}
]
```

资金流快照示例：

```json
[
  {"symbol": "NVDA", "net_inflow": 2400000, "anomaly": true},
  {"symbol": "AAPL", "net_inflow": -500000, "anomaly": false}
]
```

适配器已经拆分完成，后续可以继续升级为直接读取 OpenD / SDK 的实时数据。

## 已修复的 P0 问题

- `--refresh-prices` 现在会真正刷新 bulk price cache，不再是空转参数
- `rs_vs_sector_min` 已修正为正确的小数收益率单位 `-0.02`
- Breadth 现在会真实使用 `adv_ratio`、`nh_nl_ratio`、`% above 200EMA` 配置门槛
- 综合评分已改成和文档一致的 100 分架构，不再出现代码和注释不一致

## 风控规则

- 单笔期权亏损超过 50%：离场
- 单一标的仓位超过总资金 40%：禁止继续加仓
- Macro / Event veto：避免新开多头风险
- 时间止损：7 天没有真正启动，就减仓或退出

## 验证命令

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall pipeline src research tests
```

## 后续可升级方向

- 从快照模式升级到 Futu/moomoo OpenD 实时采集
- 用 FinBERT 或自定义金融情绪模型替换社区情绪规则层
- 扩展 earnings、guidance、诉讼、增发、并购等更细的事件标签
- 在研究层增加 rolling validation、滑点、交易成本、按 Regime 分段回测
