import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
import random
import pandas as pd
from typing import Optional

from backtester.data.loader import load_close_matrix
from backtester.core.engine import BacktestEngine
from backtester.strategies.momentum import CrossSectionalMomentum


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError(f"Config '{path}' is empty/invalid YAML.")
    return cfg


def _discover_stooq_us_tickers(root: str, seed: int = 42) -> list:
    if not root:
        raise ValueError("universe.root is empty")

    pattern = os.path.join(root, "**", "*.us.txt")
    files = glob.glob(pattern, recursive=True)

    tickers = []
    seen = set()
    for fp in files:
        name = os.path.basename(fp).lower()
        if not name.endswith(".us.txt"):
            continue
        t = name[: -len(".us.txt")].upper()
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    tickers.sort()
    rng = random.Random(int(seed))
    rng.shuffle(tickers)
    return tickers


def _count_rows_in_range_stooq_txt(filepath: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> int:
    try:
        df = pd.read_csv(filepath, header=0, usecols=[2], engine="python")
        if df.empty:
            return 0

        col = df.columns[0]
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            s = s.astype("Int64").astype(str)
        else:
            s = s.astype(str).str.strip()

        d = pd.to_datetime(s, format="%Y%m%d", errors="coerce").dropna()
        if d.empty:
            return 0

        mask = (d >= start_dt) & (d <= end_dt)
        return int(mask.sum())
    except Exception:
        return 0


def _find_local_file(nasdaq_root: str, ticker: str) -> Optional[str]:
    target = f"{ticker.lower()}.us.txt"
    matches = glob.glob(os.path.join(nasdaq_root, "**", target), recursive=True)
    return matches[0] if matches else None


def _filter_by_coverage(nasdaq_root: str, tickers: list, start: str, end: str, min_days: int, max_keep: int) -> list:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    kept = []
    n = len(tickers)
    next_pct = 10

    for i, t in enumerate(tickers, start=1):
        if n >= 20:
            pct = int((i * 100) / n)
            if pct >= next_pct:
                print(f"Filtering universe: {pct}% ({i}/{n}) | kept={len(kept)}")
                next_pct += 10

        fp = _find_local_file(nasdaq_root, t)
        if not fp:
            continue

        days = _count_rows_in_range_stooq_txt(fp, start_dt, end_dt)
        if days >= min_days:
            kept.append(t)
            if len(kept) >= max_keep:
                break

    print(f"Universe filter complete: kept {len(kept)} tickers (min_days_in_range={min_days})")
    return kept


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def _metrics(daily: np.ndarray, equity: np.ndarray):
    total_return = float(equity[-1] - 1.0)
    vol = float(np.std(daily) * np.sqrt(252))
    sharpe = float((np.mean(daily) / (np.std(daily) + 1e-12)) * np.sqrt(252))
    mdd = float(max_drawdown(equity))
    return total_return, vol, sharpe, mdd


def _mb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 * 1024.0)


def main():
    t_start_total = time.perf_counter()

    cfg = _load_config("configs/base_config.yml")

    ucfg = cfg["universe"]
    nasdaq_root = ucfg["root"]
    max_tickers = int(ucfg["max_tickers"])
    seed = int(ucfg["seed"])
    min_days = int(ucfg["min_days_in_range"])

    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    # --- Universe discovery + filter timings ---
    t0 = time.perf_counter()
    all_tickers = _discover_stooq_us_tickers(nasdaq_root, seed)
    t1 = time.perf_counter()

    tickers = _filter_by_coverage(nasdaq_root, all_tickers, start, end, min_days, max_tickers)
    t2 = time.perf_counter()

    if not tickers:
        raise RuntimeError("No tickers passed coverage filter")

    print(f"Universe selected: {len(tickers)} tickers (mode=folder)")

    # --- Data load timing ---
    t3 = time.perf_counter()
    data = load_close_matrix(
        tickers=tickers,
        start=start,
        end=end,
        nasdaq_stocks_root=nasdaq_root,
    )
    t4 = time.perf_counter()

    T, N = data.close.shape
    print(f"Data matrix: T={T} days, N={N} tickers")

    # --- Memory footprint (arrays) ---
    close_mb = _mb(data.close.nbytes)
    print("\nMemory footprint")
    print(f"close matrix: {close_mb:.2f} MB ({data.close.dtype}, shape={data.close.shape})")

    # Strategy params
    strat_cfg = cfg["strategy"]
    lookback = int(strat_cfg["lookback"])
    top_k = min(int(strat_cfg["top_k"]), len(data.tickers))

    # Costs params
    costs_cfg = cfg.get("costs", {})
    cost_bps = float(costs_cfg.get("cost_bps", 0.0))
    commission_bps = float(costs_cfg.get("commission_bps", 0.0))
    slippage_bps = float(costs_cfg.get("slippage_bps", 0.0))
    impact_k = float(costs_cfg.get("impact_k", 0.0))

    strategy = CrossSectionalMomentum(lookback=lookback, top_k=top_k)
    engine = BacktestEngine()

    # --- Engine timing ---
    t5 = time.perf_counter()
    res = engine.run(
        data.close,
        strategy,
        cost_bps=cost_bps,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        impact_k=impact_k,
    )
    t6 = time.perf_counter()

    # Memory for result arrays
    weights_mb = _mb(res.weights.nbytes) if hasattr(res, "weights") and res.weights is not None else 0.0
    turnover_mb = _mb(res.turnover.nbytes) if hasattr(res, "turnover") and res.turnover is not None else 0.0
    costs_mb = _mb(res.costs.nbytes) if hasattr(res, "costs") and res.costs is not None else 0.0
    dg_mb = _mb(res.daily_gross.nbytes) if hasattr(res, "daily_gross") else 0.0
    dn_mb = _mb(res.daily_net.nbytes) if hasattr(res, "daily_net") else 0.0

    print(f"weights:      {weights_mb:.2f} MB (shape={res.weights.shape})")
    print(f"turnover:     {turnover_mb:.4f} MB (shape={res.turnover.shape})")
    print(f"costs:        {costs_mb:.4f} MB (shape={res.costs.shape})")
    print(f"daily_gross:  {dg_mb:.4f} MB (shape={res.daily_gross.shape})")
    print(f"daily_net:    {dn_mb:.4f} MB (shape={res.daily_net.shape})")

    # --- Output timings ---
    t_total = time.perf_counter() - t_start_total
    print("\nTiming breakdown")
    print(f"discover tickers: {(t1 - t0) * 1000:.1f} ms (found {len(all_tickers)})")
    print(f"coverage filter:  {(t2 - t1) * 1000:.1f} ms (kept {len(tickers)})")
    print(f"data load:        {(t4 - t3) * 1000:.1f} ms")
    print(f"engine run:       {(t6 - t5) * 1000:.1f} ms")
    print(f"TOTAL script:     {t_total * 1000:.1f} ms")

    print("Backtest complete")
    print("Tickers (first 25):", data.tickers[:25], "...")

    tr_g, vol_g, sh_g, mdd_g = _metrics(res.daily_gross, res.equity_gross)
    tr_n, vol_n, sh_n, mdd_n = _metrics(res.daily_net, res.equity_net)

    print("\nConfig")
    print(f"lookback: {lookback}")
    print(f"top_k: {top_k}")

    print("\nCosts model")
    print(f"cost_bps (generic): {cost_bps:.2f}")
    print(f"commission_bps:    {commission_bps:.2f}")
    print(f"slippage_bps:      {slippage_bps:.2f}")
    print(f"impact_k:          {impact_k:.6f}")

    print("\nGROSS")
    print(f"Total return: {tr_g:.2%}")
    print(f"Annualized vol: {vol_g:.2%}")
    print(f"Sharpe: {sh_g:.2f}")
    print(f"Max Drawdown: {mdd_g:.2%}")

    print("\nNET")
    print(f"Total return: {tr_n:.2%}")
    print(f"Annualized vol: {vol_n:.2%}")
    print(f"Sharpe: {sh_n:.2f}")
    print(f"Max Drawdown: {mdd_n:.2%}")

    avg_turnover = float(np.mean(res.turnover))
    avg_cost_bps = float(np.mean(res.costs) * 1e4)

    print("\nTrading frictions")
    print(f"Avg daily turnover: {avg_turnover:.3f}")
    print(f"Avg daily cost: {avg_cost_bps:.2f} bps")

    plt.plot(res.equity_gross, label="Gross")
    plt.plot(res.equity_net, label="Net")
    plt.legend()
    plt.title("Equity Curve")
    plt.show()


if __name__ == "__main__":
    main()