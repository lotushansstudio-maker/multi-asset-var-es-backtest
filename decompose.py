"""Component VaR: how much of the portfolio's one-day 99% VaR comes from each asset.

Uses the asset-level EWMA covariance (lambda 0.94, normal returns), the multivariate version of the EWMA model in
models.py, so that the contributions add up exactly to the portfolio VaR. Run after download.py:   python decompose.py
(backtest.py also calls it.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from common import ROOT, WEIGHTS, portfolio_returns, write_readme

LAM, SEED, WINDOW = 0.94, 60, 500
PERIODS = {"Full test period": (None, None),
           "Global financial crisis (Sep 2008 to Mar 2009)": ("2008-09-01", "2009-03-31"),
           "COVID crash (20 Feb to 30 Apr 2020)": ("2020-02-20", "2020-04-30"),
           "Rates shock (2022)": ("2022-01-01", "2022-12-31")}
COLORS = {"SPY": "#2563eb", "TLT": "#c2410c", "GLD": "#ca8a04", "FXE": "#15803d"}


def ewma_cov(rets, lam=LAM, seed=SEED):
    """S_t = lam * S_{t-1} + (1 - lam) * r_{t-1} r_{t-1}'; S_t uses returns up to t-1 only."""
    x = rets.values
    S = np.empty((len(x), x.shape[1], x.shape[1]))
    S[0] = x[:seed].T @ x[:seed] / seed
    for t in range(1, len(x)):
        S[t] = lam * S[t - 1] + (1 - lam) * np.outer(x[t - 1], x[t - 1])
    return S


def component_var(rets, weights=WEIGHTS, a=0.99):
    """Component VaR_i = w_i (S w)_i / sigma_p * z_a. Returns components (sum = portfolio VaR), shares and SPY-TLT correlation."""
    rets = rets[list(weights.index)]
    w = weights.values
    S = ewma_cov(rets)
    Sw = S @ w
    sig = np.sqrt(np.einsum("i,ti->t", w, Sw))
    comp = pd.DataFrame(w * Sw / sig[:, None] * stats.norm.ppf(a), index=rets.index, columns=rets.columns)
    i, j = list(rets.columns).index("SPY"), list(rets.columns).index("TLT")
    corr = pd.Series(S[:, i, j] / np.sqrt(S[:, i, i] * S[:, j, j]), index=rets.index)
    return comp, comp.div(comp.sum(axis=1), axis=0), corr


def main():
    _, rets = portfolio_returns()
    comp, share, corr = component_var(rets)
    test = rets.index[WINDOW:]
    comp, share, corr = comp.loc[test], share.loc[test], corr.loc[test]
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    pd.concat({"component": comp, "share": share}, axis=1).assign(corr_spy_tlt=corr).to_csv(
        os.path.join(ROOT, "results", "component_var.csv"))

    cols = list(share.columns)
    md = ["| Period | " + " | ".join(f"{c} share" for c in cols) + " | Portfolio 99% VaR | SPY-TLT correlation |",
          "|---|" + "---|" * (len(cols) + 2)]
    rows = dict(PERIODS)
    rows[f"Latest day ({test[-1].date()})"] = (str(test[-1].date()), str(test[-1].date()))
    for name, (a, b) in rows.items():
        m = np.ones(len(test), bool) if a is None else (test >= a) & (test <= b)
        md.append(f"| {name} | " + " | ".join(f"{100 * share[c][m].mean():.0f}%" for c in cols) +
                  f" | {100 * comp[m].sum(axis=1).mean():.2f}% | {corr[m].mean():.2f} |")
    md += ["", "Shares are each asset's average contribution to the one-day 99% VaR; they add up to 100%, and a negative "
           "share means the asset was hedging the rest of the portfolio. The decomposition uses the EWMA covariance with "
           "normal returns (lambda 0.94), the asset-level version of the EWMA model above, because that is the model in "
           "which the contributions add up exactly; it inherits that model's thin tails."]
    write_readme("DECOMP", "\n".join(md))

    g = share.groupby(share.index.year)
    yearly = g.mean()[g.size() >= 20] * 100   # drop part-years with too few days (the test starts on 2007-12-31)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    pos, neg = np.zeros(len(yearly)), np.zeros(len(yearly))
    for c in cols:
        v = yearly[c].values
        bottom = np.where(v >= 0, pos, neg)
        ax.bar(yearly.index, v, bottom=bottom, color=COLORS[c], width=0.75, label=c)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_ylabel("Share of portfolio 99% VaR, %")
    ax.set_xticks(yearly.index); ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_title("Where the portfolio's 99% VaR comes from (EWMA covariance)", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "charts", "var_contrib.png"), dpi=150); plt.close(fig)
    print("component VaR: README DECOMP section, results/component_var.csv and charts/var_contrib.png updated")


if __name__ == "__main__":
    main()
