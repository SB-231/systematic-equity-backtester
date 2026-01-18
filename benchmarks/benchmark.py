import sys
from pathlib import Path
import time
import yaml

# -------------------------------------------------
# Ensure project root is on PYTHONPATH
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


def main():
    cfg = _load_config("configs/base_config.yml")

    tickers = cfg["universe"]["tickers"]
    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    strat_cfg = cfg.get("strategy", {})
    lookback = int(strat_cfg.get("lookback", 20))
    top_k_cfg = int(strat_cfg.get("top_k", 3))
    cost_bps = float(cfg.get("costs", {}).get("cost_bps", 0.0))

    engine = BacktestEngine()

    # ---------- Cold run: load + run (may build cache) ----------
    t0 = time.perf_counter()
    data = load_close_matrix(tickers, start, end)
    top_k = min(top_k_cfg, len(data.tickers))
    strategy = CrossSectionalMomentum(lookback=lookback, top_k=top_k)
    _ = engine.run(close=data.close, strategy=strategy, cost_bps=cost_bps)
    t1 = time.perf_counter()

    # ---------- Warm run: load + run again (cache hit expected) ----------
    t2 = time.perf_counter()
    data2 = load_close_matrix(tickers, start, end)
    _ = engine.run(close=data2.close, strategy=strategy, cost_bps=cost_bps)
    t3 = time.perf_counter()

    # ---------- Engine-only timing (repeated runs) ----------
    runs = 100
    t4 = time.perf_counter()
    for _ in range(runs):
        _ = engine.run(close=data.close, strategy=strategy, cost_bps=cost_bps)
    t5 = time.perf_counter()

    cold_ms = (t1 - t0) * 1000.0
    warm_ms = (t3 - t2) * 1000.0
    engine_avg_ms = ((t5 - t4) / runs) * 1000.0

    print("Benchmark Results")
    print("-----------------")
    print(f"Universe: N={len(data.tickers)}  Days: T={data.close.shape[0]}")
    print(f"Params: lookback={lookback}, top_k={top_k}, cost_bps={cost_bps:.2f}")
    print(f"Cold run (load+run): {cold_ms:.2f} ms")
    print(f"Warm run (cache+run): {warm_ms:.2f} ms")
    print(f"Engine-only avg over {runs} runs: {engine_avg_ms:.4f} ms")


if __name__ == "__main__":
    main()