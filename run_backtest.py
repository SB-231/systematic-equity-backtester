import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
import random
import pandas as pd
import hashlib
from typing import Optional, List, Dict, Tuple, Any

from backtester.data.loader import load_close_matrix
from backtester.core.engine import BacktestEngine
from backtester.strategies.momentum import CrossSectionalMomentum


# -------------------------------
# Config
# -------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError(f"Config '{path}' is empty/invalid YAML.")
    return cfg


# -------------------------------
# Universe discovery / filter
# -------------------------------
def discover_stooq_us_tickers(root: str, seed: int = 42) -> List[str]:
    if not root:
        raise ValueError("universe.root is empty")

    pattern = os.path.join(root, "**", "*.us.txt")
    files = glob.glob(pattern, recursive=True)

    tickers: List[str] = []
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


def count_rows_in_range_stooq_txt(filepath: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> int:
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


def find_local_file(nasdaq_root: str, ticker: str) -> Optional[str]:
    target = f"{ticker.lower()}.us.txt"
    matches = glob.glob(os.path.join(nasdaq_root, "**", target), recursive=True)
    return matches[0] if matches else None


def filter_by_coverage(
    nasdaq_root: str,
    tickers: List[str],
    start: str,
    end: str,
    min_days: int,
    max_keep: int,
) -> List[str]:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    kept: List[str] = []
    n = len(tickers)
    next_pct = 10

    for i, t in enumerate(tickers, start=1):
        if n >= 20:
            pct = int((i * 100) / n)
            if pct >= next_pct:
                print(f"Filtering universe: {pct}% ({i}/{n}) | kept={len(kept)}")
                next_pct += 10

        fp = find_local_file(nasdaq_root, t)
        if not fp:
            continue

        days = count_rows_in_range_stooq_txt(fp, start_dt, end_dt)
        if days >= min_days:
            kept.append(t)
            if len(kept) >= max_keep:
                break

    print(f"Universe filter complete: kept {len(kept)} tickers (min_days_in_range={min_days})")
    return kept


# -------------------------------
# Universe cache
# -------------------------------
def universe_cache_key(
    nasdaq_root: str,
    seed: int,
    max_tickers: int,
    min_days_in_range: int,
    start: str,
    end: str,
) -> str:
    payload = f"{nasdaq_root}|seed={seed}|max={max_tickers}|minDays={min_days_in_range}|{start}|{end}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def universe_cache_path(key: str) -> str:
    os.makedirs(".cache", exist_ok=True)
    return os.path.join(".cache", f"universe_{key}.txt")


def load_universe_cache(path: str) -> Optional[List[str]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers if tickers else None


def save_universe_cache(path: str, tickers: List[str]) -> None:
    with open(path, "w") as f:
        for t in tickers:
            f.write(f"{t}\n")


def select_universe(cfg: dict) -> Tuple[List[str], Dict[str, float]]:
    ucfg = cfg["universe"]
    nasdaq_root = ucfg["root"]
    max_tickers = int(ucfg["max_tickers"])
    seed = int(ucfg["seed"])
    min_days = int(ucfg["min_days_in_range"])
    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    all_tickers = discover_stooq_us_tickers(nasdaq_root, seed)
    t1 = time.perf_counter()
    timings["discover_ms"] = (t1 - t0) * 1000.0
    timings["found"] = float(len(all_tickers))

    key = universe_cache_key(nasdaq_root, seed, max_tickers, min_days, start, end)
    path = universe_cache_path(key)

    cached = load_universe_cache(path)
    if cached is not None:
        print(f"[CACHE] Loaded universe ({len(cached)} tickers) from {path}")
        timings["filter_ms"] = 0.0
        timings["universe_cache_hit"] = 1.0
        return cached, timings

    t2 = time.perf_counter()
    kept = filter_by_coverage(nasdaq_root, all_tickers, start, end, min_days, max_tickers)
    t3 = time.perf_counter()
    timings["filter_ms"] = (t3 - t2) * 1000.0
    timings["universe_cache_hit"] = 0.0

    save_universe_cache(path, kept)
    print(f"[CACHE] Saved universe ({len(kept)} tickers) to {path}")
    return kept, timings


# -------------------------------
# Metrics / reporting
# -------------------------------
def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def metrics(daily: np.ndarray, equity: np.ndarray) -> Tuple[float, float, float, float]:
    total_return = float(equity[-1] - 1.0)
    vol = float(np.std(daily) * np.sqrt(252))
    sharpe = float((np.mean(daily) / (np.std(daily) + 1e-12)) * np.sqrt(252))
    mdd = float(max_drawdown(equity))
    return total_return, vol, sharpe, mdd


def mb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 * 1024.0)


def load_data(cfg: dict, tickers: List[str]) -> Tuple[Any, float]:
    ucfg = cfg["universe"]
    nasdaq_root = ucfg["root"]
    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    t0 = time.perf_counter()
    data = load_close_matrix(
        tickers=tickers,
        start=start,
        end=end,
        nasdaq_stocks_root=nasdaq_root,
    )
    t1 = time.perf_counter()
    return data, (t1 - t0) * 1000.0


def run_engine(cfg: dict, data) -> Tuple[Any, float]:
    strat_cfg = cfg["strategy"]
    lookback = int(strat_cfg["lookback"])
    top_k = min(int(strat_cfg["top_k"]), len(data.tickers))

    costs_cfg = cfg.get("costs", {})
    cost_bps = float(costs_cfg.get("cost_bps", 0.0))
    commission_bps = float(costs_cfg.get("commission_bps", 0.0))
    slippage_bps = float(costs_cfg.get("slippage_bps", 0.0))
    impact_k = float(costs_cfg.get("impact_k", 0.0))

    strategy = CrossSectionalMomentum(lookback=lookback, top_k=top_k)
    engine = BacktestEngine()

    t0 = time.perf_counter()
    res = engine.run(
        data.close,
        strategy,
        cost_bps=cost_bps,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        impact_k=impact_k,
    )
    t1 = time.perf_counter()
    return res, (t1 - t0) * 1000.0


def print_report(cfg: dict, data, res, timings: Dict[str, float], load_ms: float, engine_ms: float, total_ms: float) -> None:
    T, N = data.close.shape
    print(f"Universe selected: requested={cfg['universe']['max_tickers']} used(N)={N}  Days(T)={T}")
    print(f"Data matrix: T={T} days, N={N} tickers")

    print("\nMemory footprint")
    print(f"close matrix: {mb(data.close.nbytes):.2f} MB ({data.close.dtype}, shape={data.close.shape})")
    print(f"weights:      {mb(res.weights.nbytes):.2f} MB (shape={res.weights.shape})")
    print(f"turnover:     {mb(res.turnover.nbytes):.4f} MB (shape={res.turnover.shape})")
    print(f"costs:        {mb(res.costs.nbytes):.4f} MB (shape={res.costs.shape})")
    print(f"daily_gross:  {mb(res.daily_gross.nbytes):.4f} MB (shape={res.daily_gross.shape})")
    print(f"daily_net:    {mb(res.daily_net.nbytes):.4f} MB (shape={res.daily_net.shape})")

    print("\nTiming breakdown")
    print(f"discover tickers: {timings.get('discover_ms', 0.0):.1f} ms (found {int(timings.get('found', 0))})")
    if timings.get("universe_cache_hit", 0.0) == 1.0:
        print("coverage filter:  0.0 ms (universe cache hit)")
    else:
        print(f"coverage filter:  {timings.get('filter_ms', 0.0):.1f} ms")
    print(f"data load:        {load_ms:.1f} ms")
    print(f"engine run:       {engine_ms:.1f} ms")
    print(f"TOTAL script:     {total_ms:.1f} ms")

    print("Backtest complete")
    print("Tickers (first 25):", data.tickers[:25], "...")

    strat_cfg = cfg["strategy"]
    lookback = int(strat_cfg["lookback"])
    top_k = min(int(strat_cfg["top_k"]), len(data.tickers))

    costs_cfg = cfg.get("costs", {})
    cost_bps = float(costs_cfg.get("cost_bps", 0.0))
    commission_bps = float(costs_cfg.get("commission_bps", 0.0))
    slippage_bps = float(costs_cfg.get("slippage_bps", 0.0))
    impact_k = float(costs_cfg.get("impact_k", 0.0))

    tr_g, vol_g, sh_g, mdd_g = metrics(res.daily_gross, res.equity_gross)
    tr_n, vol_n, sh_n, mdd_n = metrics(res.daily_net, res.equity_net)

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


def main():
    t_start_total = time.perf_counter()

    cfg = load_config("configs/base_config.yml")

    # Universe selection (cached)
    tickers, timings = select_universe(cfg)
    if not tickers:
        raise RuntimeError("No tickers selected")

    # Data load
    data, load_ms = load_data(cfg, tickers)

    # Engine run
    res, engine_ms = run_engine(cfg, data)

    total_ms = (time.perf_counter() - t_start_total) * 1000.0

    print_report(cfg, data, res, timings, load_ms, engine_ms, total_ms)


if __name__ == "__main__":
    main()