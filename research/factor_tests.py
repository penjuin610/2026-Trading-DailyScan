"""
research/factor_tests.py
─────────────────────────────────────────────────────────────
单因子测试框架

用法：
  python -m research.factor_tests --factor rs_vs_spy --forward 10
  python -m research.factor_tests --all --forward 5

输出：
  每个因子分组（高/低）的未来 N 日收益和最大回撤对比
  outputs/reports/factor_test_{factor}_{date}.csv
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# research/factor_tests.py → parents[0]=research/ → parents[1]=trading_system/ (root)
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adapters.price_client import PriceClient
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def build_factor_dataset(
    universe_prices: dict[str, pd.DataFrame],
    etf_prices:      dict[str, pd.DataFrame],
    forward_days:    int = 10,
) -> pd.DataFrame:
    """
    为每只股票的每个交易日计算所有因子值和未来收益。
    返回格式：date | symbol | [因子列] | fwd_ret_N | fwd_max_dd_N
    """
    spy_cl = etf_prices.get("SPY", pd.DataFrame()).get("close", pd.Series())
    rows = []

    for sym, df in universe_prices.items():
        col = "close" if "close" in df.columns else "Close"
        if col not in df.columns or len(df) < 250:
            continue
        cl = df[col].rename("close")

        e21  = _ema(cl, 21)
        e50  = _ema(cl, 50)
        e200 = _ema(cl, 200)

        # ATR14
        if "high" in df.columns and "low" in df.columns:
            hi = df["high"]; lo = df["low"]
        else:
            hi = cl * 1.005; lo = cl * 0.995
        tr = pd.concat([hi-lo, (hi-cl.shift(1)).abs(), (lo-cl.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()

        # Volume ratio
        vol = df.get("volume", df.get("Volume", pd.Series(dtype=float)))
        vol_ratio = vol / vol.rolling(20).mean() if not vol.empty else pd.Series(1, index=cl.index)

        # RS vs SPY
        ret20 = cl.pct_change(20)
        if spy_cl is not None and len(spy_cl) > 20:
            spy_ret20 = spy_cl.pct_change(20).reindex(cl.index).ffill()
        else:
            spy_ret20 = pd.Series(0.0, index=cl.index)
        rs_vs_spy = ret20 - spy_ret20

        # 未来 N 日收益
        fwd_ret  = cl.pct_change(forward_days).shift(-forward_days)

        # 未来 N 日最大回撤（近似）
        fwd_max_dd = pd.Series(index=cl.index, dtype=float)
        for i in range(len(cl) - forward_days):
            window = cl.iloc[i+1:i+1+forward_days]
            peak = float(cl.iloc[i])
            trough = float(window.min())
            fwd_max_dd.iloc[i] = (trough - peak) / peak if peak > 0 else 0

        for i in range(200, len(cl) - forward_days - 1):
            dt = cl.index[i]
            rows.append({
                "date":       dt,
                "symbol":     sym,
                # 因子值
                "above_200":  int(float(cl.iloc[i]) > float(e200.iloc[i])),
                "ema21_gt_50":int(float(e21.iloc[i]) > float(e50.iloc[i])),
                "ema50_slope":int(float(e50.iloc[i]) > float(e50.iloc[i-5])),
                "rs_vs_spy":  float(rs_vs_spy.iloc[i]),
                "vol_ratio":  float(vol_ratio.iloc[i]) if i < len(vol_ratio) else 1.0,
                "atr14_pct":  float(atr14.iloc[i]) / float(cl.iloc[i]) if float(cl.iloc[i]) > 0 else 0,
                "dist_21ema": (float(cl.iloc[i]) - float(e21.iloc[i])) / float(e21.iloc[i]),
                # 标签
                "fwd_ret":    float(fwd_ret.iloc[i]) if pd.notna(fwd_ret.iloc[i]) else float("nan"),
                "fwd_max_dd": float(fwd_max_dd.iloc[i]),
            })

    return pd.DataFrame(rows).dropna(subset=["fwd_ret"])


def test_single_factor(
    df:         pd.DataFrame,
    factor:     str,
    n_bins:     int = 3,
    forward_days: int = 10,
) -> pd.DataFrame:
    """
    对单个因子做分组测试，比较不同分位的未来收益分布。
    n_bins=2 → 高/低两组
    n_bins=3 → 高/中/低三组
    """
    if factor not in df.columns:
        logger.error("Factor '%s' not in dataset. Available: %s", factor, list(df.columns))
        return pd.DataFrame()

    df = df.copy()
    df["bin"] = pd.qcut(df[factor], n_bins, labels=False, duplicates="drop")

    result = df.groupby("bin").agg(
        count      = ("fwd_ret", "count"),
        mean_ret   = ("fwd_ret", "mean"),
        median_ret = ("fwd_ret", "median"),
        win_rate   = ("fwd_ret", lambda x: (x > 0).mean()),
        avg_dd     = ("fwd_max_dd", "mean"),
        sharpe_proxy = ("fwd_ret", lambda x: x.mean() / (x.std() + 1e-8)),
    ).round(4)

    result.index = [f"Q{i+1}(low)" if i==0
                    else f"Q{i+1}(high)" if i==n_bins-1
                    else f"Q{i+1}(mid)"
                    for i in range(len(result))]
    result["factor"] = factor
    result[f"fwd_days"] = forward_days

    print(f"\n{'─'*55}")
    print(f"  Factor: {factor}  |  Forward: {forward_days}d  |  N={len(df)}")
    print("─"*55)
    print(result[["count","mean_ret","win_rate","avg_dd","sharpe_proxy"]].to_string())
    return result


def run_ablation(
    df:           pd.DataFrame,
    factors:      list[str],
    forward_days: int = 10,
) -> pd.DataFrame:
    """
    组合因子 ablation：逐步加入因子，观察整体绩效变化。
    """
    print(f"\n{'═'*55}")
    print("  Ablation Study")
    print("═"*55)

    rows = []
    active_filters = []
    baseline = df.copy()

    # 基准（全部）
    base_stats = {
        "config":    "baseline_no_filter",
        "n_signals": len(baseline),
        "mean_ret":  baseline["fwd_ret"].mean(),
        "win_rate":  (baseline["fwd_ret"] > 0).mean(),
        "avg_dd":    baseline["fwd_max_dd"].mean(),
    }
    rows.append(base_stats)
    print(f"  baseline:       N={base_stats['n_signals']:5d}  "
          f"ret={base_stats['mean_ret']:+.3f}  "
          f"win={base_stats['win_rate']:.2f}  "
          f"dd={base_stats['avg_dd']:.3f}")

    for fac in factors:
        if fac not in df.columns:
            continue
        # 对连续因子：只保留 top tercile
        if df[fac].nunique() > 2:
            threshold = df[fac].quantile(0.67)
            active_filters.append(df[fac] >= threshold)
        else:
            # 二值因子：保留 = 1
            active_filters.append(df[fac] == 1)

        mask = active_filters[0].copy()
        for m in active_filters[1:]:
            mask = mask & m
        filtered = df[mask]

        stats = {
            "config":    "+".join(factors[:len(active_filters)]),
            "n_signals": len(filtered),
            "mean_ret":  filtered["fwd_ret"].mean(),
            "win_rate":  (filtered["fwd_ret"] > 0).mean(),
            "avg_dd":    filtered["fwd_max_dd"].mean(),
        }
        rows.append(stats)
        print(f"  +{fac:<20} N={stats['n_signals']:5d}  "
              f"ret={stats['mean_ret']:+.3f}  "
              f"win={stats['win_rate']:.2f}  "
              f"dd={stats['avg_dd']:.3f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor",  type=str, default="rs_vs_spy")
    parser.add_argument("--forward", type=int, default=10)
    parser.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    with open(ROOT / "config" / "universe.yaml") as f:
        universe_cfg = yaml.safe_load(f)
    stocks = universe_cfg.get("stocks", [])[:30]  # 先用前 30 只测试

    client = PriceClient()
    logger.info("Fetching prices for %d stocks...", len(stocks))
    universe_prices = client.fetch_bulk(stocks, period="2y")
    etf_prices      = client.fetch_regime_etfs()

    logger.info("Building factor dataset...")
    df = build_factor_dataset(universe_prices, etf_prices, forward_days=args.forward)
    logger.info("Dataset: %d rows, %d symbols", len(df), df["symbol"].nunique())

    if args.all:
        FACTORS = ["above_200","ema21_gt_50","ema50_slope","rs_vs_spy",
                   "vol_ratio","dist_21ema"]
        for f in FACTORS:
            test_single_factor(df, f, forward_days=args.forward)

        ablation_df = run_ablation(
            df,
            ["above_200","ema21_gt_50","ema50_slope","rs_vs_spy"],
            forward_days=args.forward,
        )
        out = REPORT_DIR / f"ablation_{args.forward}d.csv"
        ablation_df.to_csv(out, index=False)
        logger.info("Ablation saved: %s", out)
    else:
        result = test_single_factor(df, args.factor, forward_days=args.forward)
        if not result.empty:
            out = REPORT_DIR / f"factor_{args.factor}_{args.forward}d.csv"
            result.to_csv(out)
            logger.info("Saved: %s", out)
