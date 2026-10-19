import os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

# A simple balanced multi-asset portfolio, rebalanced daily to fixed weights.
ASSETS = {"SPY": "US equities", "TLT": "US Treasuries 20y+", "GLD": "Gold", "FXE": "Euro vs US dollar"}
WEIGHTS = pd.Series({"SPY": 0.40, "TLT": 0.40, "GLD": 0.10, "FXE": 0.10})


def portfolio_returns():
    """Daily simple returns of the fixed-weight portfolio, using days when all four ETFs traded."""
    px = pd.read_csv(os.path.join(DATA, "adjclose_daily.csv"), index_col=0, parse_dates=True)
    rets = px[list(WEIGHTS.index)].pct_change().dropna(how="any")
    return rets @ WEIGHTS, rets


def write_readme(section, md):
    """Replace the text between <!-- SECTION --> markers in README.md with fresh results."""
    path = os.path.join(ROOT, "README.md")
    s = open(path, encoding="utf-8").read()
    s = re.sub(rf"(<!-- {section} -->).*?(<!-- /{section} -->)", lambda m: f"{m.group(1)}\n{md}\n{m.group(2)}", s, flags=re.S)
    open(path, "w", encoding="utf-8").write(s)
