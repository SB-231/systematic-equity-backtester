from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketData:
    dates: np.ndarray      # shape [T]
    tickers: List[str]     # length N
    close: np.ndarray      # shape [T, N]


def _print_progress(i: int, n: int, next_pct: int) -> int:
    """
    Print progress every 10% while loading tickers.
    Returns the next percentage threshold to print.
    """
    # For small N, percent printing isn't helpful (too noisy / trivial)
    if n < 20:
        return next_pct

    pct = int((i * 100) / n)
    if pct >= next_pct:
        print(f"Loading stocks: {pct}% ({i}/{n})")
        return next_pct + 10
    return next_pct


def _candidate_filenames(ticker: str) -> List[str]:
    """
    Local Stooq US stock files:
      - aapl.us.txt
      - for dot tickers: brk-b.us.txt
    """
    t = ticker.strip().lower()
    names = [f"{t}.us.txt"]
    if "." in t:
        names.append(f"{t.replace('.', '-')}.us.txt")
    return names


def _read_stooq_txt(path: str) -> pd.DataFrame:
    """
    Robust reader for Stooq TXT files like:

      <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
      AACB.US,D,20250407,000000,10.1,10.1,9.88,9.88,8365,0
      ...

    Handles:
      - delimiter guess: ',', ';', or tab
      - column names with <> and casing
      - DATE stored as YYYYMMDD (e.g. 20250407)
    Requires at least DATE + CLOSE (or Date + Close after normalization).
    """
    # Guess delimiter from first line
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    candidates = [
        (",", first_line.count(",")),
        (";", first_line.count(";")),
        ("\t", first_line.count("\t")),
    ]
    delimiter = max(candidates, key=lambda x: x[1])[0]

    df = pd.read_csv(path, sep=delimiter)
    if df.empty:
        raise ValueError("Empty/invalid stooq txt")

    # Normalize column names: strip, remove <>, keep original case but match case-insensitively
    rename_map = {}
    for c in df.columns:
        clean = str(c).strip().replace("<", "").replace(">", "")
        rename_map[c] = clean
    df = df.rename(columns=rename_map)

    # Case-insensitive lookup
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # Stooq may have DATE/CLOSE or Date/Close
    date_col = cols_lower.get("date")
    close_col = cols_lower.get("close")

    if date_col is None or close_col is None:
        raise ValueError("Empty/invalid stooq txt (missing DATE/CLOSE columns)")

    # Rename to standard names
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})

    # Parse dates: many Stooq dumps use YYYYMMDD integers/strings
    # Try strict YYYYMMDD first, then fallback.
    s = df["Date"]

    # If numeric, convert to string without decimals
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("Int64").astype(str)
    else:
        s = s.astype(str).str.strip()

    # Parse YYYYMMDD where possible
    df["Date"] = pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close"])
    if df.empty:
        raise ValueError("No valid rows after parsing (DATE/CLOSE)")

    df = df.sort_values("Date").set_index("Date")
    df = df.sort_index()
    return df


def _cache_key(
    tickers: List[str],
    start: str,
    end: str,
    ffill_limit: int,
    min_ticker_coverage: float,
    min_row_coverage: float,
    nasdaq_stocks_root: str,
) -> str:
    payload = (
        "|".join(sorted(tickers))
        + f"||{start}||{end}"
        + f"||ffill={ffill_limit}"
        + f"||minTicker={min_ticker_coverage}"
        + f"||minRow={min_row_coverage}"
        + f"||root={nasdaq_stocks_root}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_paths(key: str) -> Tuple[str, str]:
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    return (
        os.path.join(cache_dir, f"close_{key}.npz"),
        os.path.join(cache_dir, f"close_{key}.meta.txt"),
    )


def _find_shard_dirs(nasdaq_stocks_root: str) -> List[str]:
    """
    Stooq shards into numeric folders: 1,2,3,...
    """
    if not os.path.isdir(nasdaq_stocks_root):
        raise FileNotFoundError(f"nasdaq_stocks root not found: {nasdaq_stocks_root}")

    shards = []
    for name in os.listdir(nasdaq_stocks_root):
        full = os.path.join(nasdaq_stocks_root, name)
        if os.path.isdir(full) and name.isdigit():
            shards.append(full)

    shards.sort(key=lambda p: int(os.path.basename(p)))
    if not shards:
        raise FileNotFoundError(f"No shard folders found under {nasdaq_stocks_root}")

    return shards


def load_close_matrix(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
    ffill_limit: int = 5,
    min_ticker_coverage: float = 0.90,
    min_row_coverage: float = 0.98,
    nasdaq_stocks_root: str = "data_raw/stooq_us_daily/daily/us/nasdaq_stocks",
) -> MarketData:
    """
    Local-only loader (development mode).
    Uses ONLY NASDAQ stocks:
      nasdaq_stocks_root/{1,2,3,...}/*.us.txt

    Progress:
      - Prints progress at 10%, 20%, ..., 100% (for N >= 20)
    """
    shard_dirs = _find_shard_dirs(nasdaq_stocks_root)

    key = _cache_key(
        tickers,
        start,
        end,
        ffill_limit,
        min_ticker_coverage,
        min_row_coverage,
        nasdaq_stocks_root,
    )
    npz_path, meta_path = _cache_paths(key)

    if use_cache and os.path.exists(npz_path):
        arr = np.load(npz_path, allow_pickle=True)
        print(f"[CACHE] Loaded {arr['close'].shape} from {npz_path}")
        return MarketData(
            dates=arr["dates"],
            tickers=arr["tickers"].tolist(),
            close=arr["close"],
        )

    frames: List[pd.DataFrame] = []
    good_tickers: List[str] = []

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    total = len(tickers)
    next_pct = 10

    for i, ticker in enumerate(tickers, start=1):
        # Progress indicator
        next_pct = _print_progress(i, total, next_pct)

        loaded = False
        last_err = None

        for fname in _candidate_filenames(ticker):
            for shard in shard_dirs:
                path = os.path.join(shard, fname)
                if not os.path.exists(path):
                    continue
                try:
                    df = _read_stooq_txt(path)

                    df_range = df.loc[start_dt:end_dt]
                    if df_range.empty:
                        dmin = df.index.min()
                        dmax = df.index.max()
                        last_err = f"no rows in range (file range {dmin.date()}..{dmax.date()})"
                        continue

                    frames.append(df_range[["Close"]].rename(columns={"Close": ticker}))
                    good_tickers.append(ticker)
                    loaded = True
                    break
                except Exception as e:
                    last_err = str(e)

            if loaded:
                break

        if not loaded:
            print(f"[WARN] skipped {ticker}: {last_err or 'file not found'}")

    if not frames:
        raise RuntimeError(
            "No local data loaded. "
            "Check nasdaq_stocks_root and date range in config."
        )

    merged = pd.concat(frames, axis=1).sort_index()

    # Drop sparse tickers
    coverage = merged.notna().mean(axis=0)
    keep_cols = coverage[coverage >= min_ticker_coverage].index.tolist()
    merged = merged[keep_cols]

    # Fill small gaps
    merged = merged.ffill(limit=ffill_limit)

    # Keep rows with enough cross-sectional data
    required = int(np.ceil(min_row_coverage * merged.shape[1]))
    merged = merged.loc[merged.notna().sum(axis=1) >= required]

    # Dense matrix
    merged = merged.dropna()

    print(f"Loaded {len(good_tickers)}/{total} tickers. Final matrix: {merged.shape}")

    dates = merged.index.to_numpy()
    close = merged.to_numpy(dtype=np.float64)
    final_tickers = merged.columns.tolist()

    np.savez_compressed(
        npz_path,
        dates=dates,
        close=close,
        tickers=np.array(final_tickers, dtype=object),
    )
    with open(meta_path, "w") as f:
        f.write(f"shape={close.shape}\n")
        f.write(f"start={start}\nend={end}\n")

    print(f"[CACHE] Saved to {npz_path}")
    return MarketData(dates=dates, tickers=final_tickers, close=close)