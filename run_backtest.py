import yaml
from backtester.data.loader import load_close_matrix


def main():
    with open("configs/base_config.yml") as f:
        cfg = yaml.safe_load(f)

    tickers = cfg["universe"]["tickers"]
    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    data = load_close_matrix(tickers, start, end)

    print("Loaded market data")
    print("Tickers:", data.tickers)
    print("Dates:", data.dates.shape)
    print("Close matrix shape:", data.close.shape)


if __name__ == "__main__":
    main()