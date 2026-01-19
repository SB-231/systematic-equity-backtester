import time
import yaml

from backtester.universe import select_universe
from backtester.data.loader import load_close_matrix
from backtester.core.engine import BacktestEngine
from backtester.strategies.momentum import CrossSectionalMomentum


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError(f"Config '{path}' is empty/invalid YAML.")
    return cfg


def _mb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 * 1024.0)


def main():
    cfg = _load_config("configs/base_config.yml")

    # -------------------------------------------------
    # Universe selection (shared + cached)
    # -------------------------------------------------
    t0 = time.perf_counter()
    tickers, timings = select_universe(cfg)
    t1 = time.perf_counter()

    if not tickers:
        raise RuntimeError("No tickers selected")

    requested = int(cfg["universe"]["max_tickers"])
    nasdaq_root = cfg["universe"]["root"]

    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    # -------------------------------------------------
    # Strategy params
    # -------------------------------------------------
    strat_cfg = cfg.get("strategy", {})
    lookback = int(strat_cfg.get("lookback", 20))
    top_k_cfg = int(strat_cfg.get("top_k", 3))

    # -------------------------------------------------
    # Costs params
    # -------------------------------------------------
    costs_cfg = cfg.get("costs", {})
    cost_bps = float(costs_cfg.get("cost_bps", 0.0))
    commission_bps = float(costs_cfg.get("commission_bps", 0.0))
    slippage_bps = float(costs_cfg.get("slippage_bps", 0.0))
    impact_k = float(costs_cfg.get("impact_k", 0.0))

    engine = BacktestEngine()

    # -------------------------------------------------
    # Cold run: load + run (may build cache)
    # -------------------------------------------------
    t2 = time.perf_counter()
    data = load_close_matrix(
        tickers=tickers,
        start=start,
        end=end,
        nasdaq_stocks_root=nasdaq_root,
    )
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

    # -------------------------------------------------
    # Warm run: cache hit expected
    # -------------------------------------------------
    t4 = time.perf_counter()
    data2 = load_close_matrix(
        tickers=tickers,
        start=start,
        end=end,
        nasdaq_stocks_root=nasdaq_root,
    )
    _ = engine.run(
        close=data2.close,
        strategy=strategy,
        cost_bps=cost_bps,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        impact_k=impact_k,
    )
    t5 = time.perf_counter()

    # -------------------------------------------------
    # Engine-only benchmark
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Universe timing (prefer internal cache metrics)
    # -------------------------------------------------
    discover_ms = float(timings.get("discover_ms", 0.0))
    filter_ms = float(timings.get("filter_ms", 0.0))
    found = int(timings.get("found", 0))
    cache_hit = bool(timings.get("universe_cache_hit", 0.0) == 1.0)

    if discover_ms <= 0.0 and filter_ms <= 0.0:
        universe_ms = (t1 - t0) * 1000.0
    else:
        universe_ms = discover_ms + filter_ms

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    print("\nBenchmark Results")
    print("-----------------")
    print(
        f"Universe: requested={requested}  "
        f"used(N)={len(data.tickers)}  "
        f"Days(T)={data.close.shape[0]}"
    )

    if cache_hit:
        print(
            f"Universe selection (discover+filter): "
            f"{universe_ms:.2f} ms (found={found}, cache_hit=True)"
        )
    else:
        print(
            f"Universe selection (discover+filter): "
            f"{universe_ms:.2f} ms (found={found})"
        )

    print(f"Params: lookback={lookback}, top_k={top_k}")
    print(
        "Costs: "
        f"cost_bps={cost_bps:.2f}, "
        f"commission_bps={commission_bps:.2f}, "
        f"slippage_bps={slippage_bps:.2f}, "
        f"impact_k={impact_k:.6f}"
    )
    print(f"Cold run (load+run): {cold_ms:.2f} ms")
    print(f"Warm run (cache+run): {warm_ms:.2f} ms")
    print(f"Engine-only avg over {runs} runs: {engine_avg_ms:.4f} ms")

    print("\nMemory (data close matrix)")
    print(
        f"close: {_mb(data.close.nbytes):.2f} MB "
        f"(dtype={data.close.dtype}, shape={data.close.shape})"
    )


if __name__ == "__main__":
    main()