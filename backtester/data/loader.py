from __future__ import annotations

import io
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MarketData:
    dates: np.ndarray      # shape [T]
    tickers: List[str]     # length N
    close: np.ndarray      # shape [T, N]


def _stooq_symbol(ticker: str) -> str:
    return f"{ticker.lower()}.us"


def _download_stooq_daily(ticker: str) -> pd.DataFrame:
    symbol = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    return df


def load_close_matrix(
    tickers: List[str],
    start: str,
    end: str,
) -> MarketData:
    frames = []
    valid_tickers = []

    for t in tickers:
        try:
            df = _download_stooq_daily(t)
            df = df.loc[start:end]

            if df.empty:
                continue

            frames.append(df[["Close"]].rename(columns={"Close": t}))
            valid_tickers.append(t)

        except Exception as e:
            print(f"[WARN] Failed {t}: {e}")

    if not frames:
        raise RuntimeError("No data loaded")

    merged = pd.concat(frames, axis=1).sort_index()

    # Handle missing data safely
    merged = merged.ffill(limit=5).dropna()

    return MarketData(
        dates=merged.index.to_numpy(),
        tickers=valid_tickers,
        close=merged.to_numpy(dtype=np.float64),
    )