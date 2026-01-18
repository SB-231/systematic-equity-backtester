import sys
from pathlib import Path
import time
import os
import glob
import random
from typing import Optional

import yaml
import numpy as np
import pandas as pd

# -------------------------------------------------
# Ensure project root is on PYTHONPATH
# (Packaging will remove the need for this later)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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


def _mb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 * 1024.0)


def main():
    cfg = _load_config("configs/base_config.yml")

    # --- Universe (folder-mode + coverage filter) ---
    ucfg = cfg["universe"]
    nasdaq_root = ucfg["root"]
    max_tickers = int(ucfg["max_tickers"])
    seed = int(ucfg["seed"])
    min_days = int(ucfg["min_days_in_range"])

    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    t0 = time.perf_counter()
    all_tickers = _discover_stooq_us_tickers(nasdaq_root, seed)
    tickers = _filter_by_coverage(nasdaq_root, all_tickers, start, end, min_days, max_tickers)
    if not tickers:
        raise RuntimeError("No tickers passed coverage filter")
    t1 = time.perf_counter()

    # --- Strategy params ---
    strat_cfg = cfg.get("strategy", {})
    lookback = int(strat_cfg.get("lookback", 20))
    top_k_cfg = int(strat_cfg.get("top_k", 3))

    # --- Costs params (full model) ---
    costs_cfg = cfg.get("costs", {})
    cost_bps = float(costs_cfg.get("cost_bps", 0.0))
    commission_bps = float(costs_cfg.get("commission_bps", 0.0))
    slippage_bps = float(costs_cfg.get("slippage_bps", 0.0))
    impact_k = float(costs_cfg.get("impact_k", 0.0))

    engine = BacktestEngine()

    # ---------- Cold run: load + run (may build cache) ----------
    t2 = time.perf_counter()
    data = load_close_matrix(tickers, start, end, nasdaq_stocks_root=nasdaq_root)
    top_k = min(top_k_cfg, len(data.tickers))
    strategy = CrossSectionalMomentum(lookback=lookback, top_k=top_k)
    _ = engine.run(
        close=data.close,
        strategy=strategy,
        cost_bps=cost_bps,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        impact_k=impact_k,
    )
    t3 = time.perf_counter()

    # ---------- Warm run: load + run again (cache hit expected) ----------
    t4 = time.perf_counter()
    data2 = load_close_matrix(tickers, start, end, nasdaq_stocks_root=nasdaq_root)
    _ = engine.run(
        close=data2.close,
        strategy=strategy,
        cost_bps=cost_bps,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        impact_k=impact_k,
    )
    t5 = time.perf_counter()

    # ---------- Engine-only timing (repeated runs) ----------
    runs = 100
    t6 = time.perf_counter()
    for _ in range(runs):
        _ = engine.run(
            close=data.close,
            strategy=strategy,
            cost_bps=cost_bps,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            impact_k=impact_k,
        )
    t7 = time.perf_counter()

    cold_ms = (t3 - t2) * 1000.0
    warm_ms = (t5 - t4) * 1000.0
    engine_avg_ms = ((t7 - t6) / runs) * 1000.0
    universe_ms = (t1 - t0) * 1000.0

    print("\nBenchmark Results")
    print("-----------------")
    print(f"Universe: requested={len(tickers)}  used(N)={len(data.tickers)}  Days(T)={data.close.shape[0]}")
    print(f"Universe selection (discover+filter): {universe_ms:.2f} ms (found={len(all_tickers)})")
    print(f"Params: lookback={lookback}, top_k={top_k}")
    print("Costs: "
          f"cost_bps={cost_bps:.2f}, commission_bps={commission_bps:.2f}, "
          f"slippage_bps={slippage_bps:.2f}, impact_k={impact_k:.6f}")
    print(f"Cold run (load+run): {cold_ms:.2f} ms")
    print(f"Warm run (cache+run): {warm_ms:.2f} ms")
    print(f"Engine-only avg over {runs} runs: {engine_avg_ms:.4f} ms")

    print("\nMemory (data close matrix)")
    print(f"close: {_mb(data.close.nbytes):.2f} MB (dtype={data.close.dtype}, shape={data.close.shape})")


if __name__ == "__main__":
    main()