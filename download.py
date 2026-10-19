"""Download daily adjusted prices for the four portfolio ETFs from Yahoo Finance into data/.

Run once:   python download.py
"""
import os
import pandas as pd
import yfinance as yf
from common import ASSETS, DATA

START = "2006-01-01"   # FXE, the youngest ETF, starts in December 2005


def main():
    os.makedirs(DATA, exist_ok=True)
    adj = {}
    for t in ASSETS:
        h = yf.Ticker(t).history(start=START, interval="1d", auto_adjust=False)
        if h.empty:
            raise SystemExit(f"No data for {t}; check your connection and try again.")
        h.index = h.index.tz_localize(None).normalize()
        adj[t] = h["Adj Close"]
        print(f"ok   {t:4s} {len(h):5d} days  {h.index[0].date()} to {h.index[-1].date()}")
    pd.DataFrame(adj).to_csv(os.path.join(DATA, "adjclose_daily.csv"))
    print("saved data/adjclose_daily.csv")


if __name__ == "__main__":
    main()
