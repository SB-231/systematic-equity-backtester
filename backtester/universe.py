from __future__ import annotations

import os
import glob
import random
import hashlib
import time
from typing import Optional, List, Dict, Tuple

import pandas as pd


# -------------------------------
# Universe discovery / filter
# -------------------------------
def discover_stooq_us_tickers(root: str, seed: int = 42) -> List[str]:
    """
    Discover tickers by scanning local Stooq *.us.txt files under `root`.
    Returns a shuffled list (stable by seed).
    """
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
    """
    Coverage estimation: reads only the DATE column from Stooq TXT
    and counts how many rows fall in [start_dt, end_dt].
    Assumes Stooq format where date is the 3rd column (index 2).
    """
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
    """
    Find the local file for a ticker under the NASDAQ root shard folders.
    """
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
    """
    Keep tickers that have >= min_days within [start, end], stop after max_keep.
    Prints progress at 10% increments for large universes.
    """
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for t in tickers:
            f.write(f"{t}\n")


def select_universe(cfg: dict) -> Tuple[List[str], Dict[str, float]]:
    """
    Shared universe selection that both run_backtest and benchmark can use.

    Returns:
      tickers: selected list
      timings: dict with discover_ms, filter_ms, found, universe_cache_hit
    """
    ucfg = cfg["universe"]
    nasdaq_root = ucfg["root"]
    max_tickers = int(ucfg["max_tickers"])
    seed = int(ucfg["seed"])
    min_days = int(ucfg["min_days_in_range"])
    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    timings: Dict[str, float] = {}

    # Discover
    t0 = time.perf_counter()
    all_tickers = discover_stooq_us_tickers(nasdaq_root, seed)
    t1 = time.perf_counter()
    timings["discover_ms"] = (t1 - t0) * 1000.0
    timings["found"] = float(len(all_tickers))

    # Cache
    key = universe_cache_key(nasdaq_root, seed, max_tickers, min_days, start, end)
    path = universe_cache_path(key)

    cached = load_universe_cache(path)
    if cached is not None:
        print(f"[CACHE] Loaded universe ({len(cached)} tickers) from {path}")
        timings["filter_ms"] = 0.0
        timings["universe_cache_hit"] = 1.0
        return cached, timings

    # Filter
    t2 = time.perf_counter()
    kept = filter_by_coverage(nasdaq_root, all_tickers, start, end, min_days, max_tickers)
    t3 = time.perf_counter()
    timings["filter_ms"] = (t3 - t2) * 1000.0
    timings["universe_cache_hit"] = 0.0

    save_universe_cache(path, kept)
    print(f"[CACHE] Saved universe ({len(kept)} tickers) to {path}")
    return kept, timings