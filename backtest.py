"""Forecast one-day VaR and ES for the portfolio with three models, then backtest the forecasts.

Run after download.py:   python backtest.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from common import ROOT, ASSETS, WEIGHTS, portfolio_returns, write_readme
import models
import decompose

WINDOW = 500          # history needed before the first forecast (about two years)
STRESS = {"Global financial crisis (Sep 2008 to Mar 2009)": ("2008-09-01", "2009-03-31"),
          "COVID crash (20 Feb to 30 Apr 2020)": ("2020-02-20", "2020-04-30"),
          "Rates shock (2022)": ("2022-01-01", "2022-12-31")}
SUBPERIODS = {"2008 to 2018 (selection)": ("2008-01-01", "2018-12-31"),
              "2019 to 2026-10-16 (hold-out)": ("2019-01-01", "2026-10-16")}
NAMES = {"hs": "Historical simulation (500 days)", "ewma": "EWMA, lambda 0.94 (normal)", "garch": "GARCH(1,1), Student-t",
         "gjr": "GJR-GARCH, Student-t", "fhs": "Filtered historical simulation (GJR)"}
SHORT = {"hs": "historical simulation", "ewma": "EWMA", "garch": "GARCH", "gjr": "GJR-GARCH", "fhs": "filtered HS"}
COLORS = {"hs": "#c2410c", "ewma": "#2563eb", "garch": "#15803d", "gjr": "#7c3aed", "fhs": "#db2777"}
MARKERS = {"hs": "o", "ewma": "s", "garch": "^", "gjr": "D", "fhs": "v"}
SHIFT = {k: pd.Timedelta(hours=h) for k, h in zip(NAMES, (-16, -8, 0, 8, 16))}


# ---------- backtests ----------
def kupiec(hits, p):
    """Proportion-of-failures test: is the breach rate equal to p?"""
    n, x = len(hits), int(hits.sum())
    pi = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - pi) + x * np.log(pi) if 0 < x < n else 0.0
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen(hits):
    """Independence test: is a breach more likely the day after a breach?"""
    h = hits.astype(int).values
    prev, cur = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (cur == 0)); n01 = np.sum((prev == 0) & (cur == 1))
    n10 = np.sum((prev == 1) & (cur == 0)); n11 = np.sum((prev == 1) & (cur == 1))
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / len(cur)
    ll = lambda k, q: k * np.log(q) if k > 0 else 0.0
    ll0 = ll(n00 + n10, 1 - p) + ll(n01 + n11, p)
    ll1 = ll(n00, 1 - p01) + ll(n01, p01) + ll(n10, 1 - p11) + ll(n11, p11)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1), n11


def dq_test(hits, var, p, lags=4):
    """Engle-Manganelli (2004) dynamic quantile test: regress hit_t - p on a constant, the last `lags` hits and
    today's VaR. If the model is right nothing should predict a breach, so all coefficients are zero."""
    h = hits.astype(float).values - p
    X = np.column_stack([np.ones(len(h) - lags)] + [h[lags - k:len(h) - k] for k in range(1, lags + 1)] + [var.values[lags:]])
    y = h[lags:]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    dq = b @ X.T @ X @ b / (p * (1 - p))
    return dq, 1 - stats.chi2.cdf(dq, X.shape[1])


def acerbi_z2(loss, var, es, a=0.975):
    """Acerbi-Szekely (2014) Z2 test for ES, written for losses. Expected value 0 if ES is right;
    positive means losses beyond VaR were worse than forecast. Their approximate thresholds:
    0.70 (5% significance) and 1.8 (0.01%)."""
    hit = loss > var
    return (loss[hit] / es[hit]).sum() / (len(loss) * (1 - a)) - 1


def traffic_light(hits, window=250):
    """Basel zones on rolling 250-day breach counts at 99%: green 0-4, yellow 5-9, red 10+."""
    c = hits.astype(int).rolling(window).sum().dropna()
    return (c <= 4).mean(), ((c >= 5) & (c <= 9)).mean(), (c >= 10).mean(), c


def evaluate(loss, f):
    """All backtests for one model's forecasts `f` (aligned with `loss`)."""
    h99, h975 = loss > f["VaR0.99"], loss > f["VaR0.975"]
    lr_uc, p_uc = kupiec(h99, 0.01)
    lr_ind, p_ind, clusters = christoffersen(h99)
    p_cc = 1 - stats.chi2.cdf(lr_uc + lr_ind, 2)
    g, y, r, _ = traffic_light(h99)
    _, p_dq = dq_test(h99, f["VaR0.99"], 0.01)
    return {"breaches": int(h99.sum()), "expected": len(loss) * 0.01, "rate": h99.mean(),
            "kupiec_p": p_uc, "ind_p": p_ind, "cc_p": p_cc, "dq_p": p_dq, "back_to_back": clusters,
            "green": g, "yellow": y, "red": r,
            "z2": acerbi_z2(loss, f["VaR0.975"], f["ES0.975"]), "breaches975": int(h975.sum()),
            "avg_var": f["VaR0.99"].mean(), "avg_es": f["ES0.975"].mean()}


# ---------- formatting ----------
pct = lambda x, d=2: f"{100 * x:.{d}f}%"
pv = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"


def verdict(e):
    if e["kupiec_p"] < 0.05 and e["rate"] > 0.01:
        return "too many breaches"
    if e["kupiec_p"] < 0.05:
        return "too conservative"
    if e["ind_p"] < 0.05 or e["dq_p"] < 0.05:
        return "right rate, but breaches cluster"
    return "passes"


def main():
    r, rets = portfolio_returns()
    loss = -r
    fc = {"hs": models.historical(loss, WINDOW), "ewma": models.ewma(r)}
    fc["garch"], params = models.garch(r, start=WINDOW)
    fc["gjr"], params_gjr = models.garch(r, start=WINDOW, asym=True)
    fc["fhs"], _ = models.garch(r, start=WINDOW, asym=True, fhs=True)
    test = loss.index[WINDOW:]
    loss_t = loss.loc[test]
    fc = {k: v.loc[test] for k, v in fc.items()}
    assert all(v.notna().all().all() for v in fc.values()), "missing forecasts in the test period"

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "charts"), exist_ok=True)
    pd.concat({k: v for k, v in fc.items()}, axis=1).assign(loss=loss_t).to_csv(os.path.join(ROOT, "results", "forecasts.csv"))
    params.to_csv(os.path.join(ROOT, "results", "garch_params.csv"))
    params_gjr.to_csv(os.path.join(ROOT, "results", "gjr_params.csv"))

    ev = {k: evaluate(loss_t, fc[k]) for k in fc}
    bt = pd.DataFrame(ev).T
    bt[["breaches", "back_to_back", "breaches975"]] = bt[["breaches", "back_to_back", "breaches975"]].astype(int)
    bt.to_csv(os.path.join(ROOT, "results", "backtests.csv"))

    # ---------- main table ----------
    n = len(test)
    lines = [f"Test period: **{test[0].date()} to {test[-1].date()}**, {n:,} trading days. "
             f"Portfolio: " + ", ".join(f"{int(WEIGHTS[t]*100)}% {t} ({ASSETS[t]})" for t in WEIGHTS.index) +
             ", rebalanced daily. Expected 99% VaR breaches: **{:.0f}**.".format(n * 0.01), "",
             "| | " + " | ".join(NAMES[k] for k in fc) + " |", "|---|" + "---|" * len(fc)]
    rows = [("99% VaR breaches (expected {:.0f})".format(n * 0.01), lambda e: f"{e['breaches']} ({pct(e['rate'])})"),
            ("Kupiec p-value (breach rate = 1%?)", lambda e: pv(e["kupiec_p"])),
            ("Christoffersen independence p-value", lambda e: pv(e["ind_p"])),
            ("Conditional coverage p-value", lambda e: pv(e["cc_p"])),
            ("Engle-Manganelli dynamic quantile p-value", lambda e: pv(e["dq_p"])),
            ("Back-to-back breaches", lambda e: str(e["back_to_back"])),
            ("Share of 250-day windows in Basel red zone (10+ breaches)", lambda e: pct(e["red"], 1)),
            ("97.5% ES test, Acerbi-Szekely Z2 (0 = right; >0.70 rejects)", lambda e: f"{e['z2']:.2f}"),
            ("Average 99% VaR (one-day loss)", lambda e: pct(e["avg_var"])),
            ("Average 97.5% ES", lambda e: pct(e["avg_es"])),
            ("Verdict on 99% VaR", verdict)]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(ev[k]) for k in fc) + " |")
    p_last = params.iloc[-1]
    lines += ["", f"Latest GARCH fit ({params.index[-1].date()}): alpha {p_last.alpha:.3f}, beta {p_last.beta:.3f}, "
              f"persistence {p_last.alpha + p_last.beta:.3f}, Student-t degrees of freedom {p_last.nu:.1f}. "
              f"Latest GJR-GARCH fit: alpha {params_gjr.iloc[-1].alpha:.3f}, gamma (extra effect of a loss) {params_gjr.iloc[-1].gamma:.3f}, "
              f"beta {params_gjr.iloc[-1].beta:.3f}."]
    # sub-periods: GJR and filtered HS were added after the first results, so 2019 onwards is the fairer test
    lines += ["", "**By sub-period.** GJR-GARCH and filtered historical simulation were added after the first results "
              "(three models, full period) had been seen, so 2008 to 2018 is in effect the period the models were chosen on "
              "and 2019 onwards is the fairer, hold-out test.", "",
              "| Period | Days | Expected | " + " | ".join(NAMES[k] for k in fc) + " |", "|---|---|---|" + "---|" * len(fc)]
    for label, (a, b) in SUBPERIODS.items():
        m = (loss_t.index >= a) & (loss_t.index <= b)
        cells = []
        for k in fc:
            h = loss_t[m] > fc[k]["VaR0.99"][m]
            cells.append(f"{int(h.sum())} (Kupiec {pv(kupiec(h, 0.01)[1])}, ind. {pv(christoffersen(h)[1])})")
        lines.append(f"| {label} | {m.sum()} | {m.sum() * 0.01:.0f} | " + " | ".join(cells) + " |")
    write_readme("RESULTS", "\n".join(lines))

    # ---------- stress periods ----------
    sl = ["| Period | Days | Expected | " + " | ".join(NAMES[k] for k in fc) + " |", "|---|---|---|" + "---|" * len(fc)]
    for name, (a, b) in STRESS.items():
        m = (loss_t.index >= a) & (loss_t.index <= b)
        cells = []
        for k in fc:
            hits = (loss_t[m] > fc[k]["VaR0.99"][m])
            cells.append(f"{int(hits.sum())}")
        sl.append(f"| {name} | {m.sum()} | {m.sum() * 0.01:.1f} | " + " | ".join(cells) + " |")
    worst = loss_t.idxmax()
    sl += ["", f"Worst day in the test: **{worst.date()}**, a loss of {pct(loss_t[worst])}. "
           "Forecast 99% VaR that morning: " + ", ".join(f"{SHORT[k]} {pct(fc[k]['VaR0.99'][worst])}" for k in fc) + "."]
    write_readme("STRESS", "\n".join(sl))

    # ---------- robustness ----------
    rb = ["| Variant | 99% VaR breaches | Kupiec p | Independence p | Z2 |", "|---|---|---|---|---|"]
    variants = {"Historical simulation, 250 days": models.historical(loss, 250),
                "Historical simulation, 1,000 days": models.historical(loss, 1000),
                "EWMA, lambda 0.97": models.ewma(r, 0.97)}
    for name, f in variants.items():
        f = f.loc[test]
        ok = f.notna().all(axis=1)
        e = evaluate(loss_t[ok], f[ok])
        rb.append(f"| {name}{' (from ' + str(ok.idxmax().date()) + ')' if not ok.all() else ''} | {e['breaches']} of {ok.sum() * 0.01:.0f} | {pv(e['kupiec_p'])} | {pv(e['ind_p'])} | {e['z2']:.2f} |")
    for k in fc:
        h95 = loss_t > fc[k]["VaR0.95"]
        rb.append(f"| {NAMES[k]}, 95% VaR | {int(h95.sum())} of {n * 0.05:.0f} | {pv(kupiec(h95, 0.05)[1])} | {pv(christoffersen(h95)[1])} | |")
    write_readme("ROBUST", "\n".join(rb))

    # ---------- charts ----------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.vlines(loss_t.index, 0, loss_t.clip(lower=0) * 100, color="#6b7280", lw=0.6, label="Daily loss")
    for k in ("hs", "garch", "fhs"):   # EWMA and GJR track GARCH closely; five lines would be unreadable
        ax.plot(fc[k].index, fc[k]["VaR0.99"] * 100, lw=0.8, alpha=0.9, color=COLORS[k], label=f"99% VaR: {NAMES[k]}")
    gh = loss_t > fc["garch"]["VaR0.99"]
    ax.scatter(loss_t[gh].index, loss_t[gh] * 100, s=8, color="black", zorder=3, label="Breach of GARCH 99% VaR")
    ax.set_ylabel("One-day loss, %"); ax.legend(frameon=False, loc="upper right")
    ax.set_title("Daily portfolio losses and one-day 99% VaR forecasts (three of the five models)", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "charts", "var_full.png"), dpi=150); plt.close(fig)

    zoom = (loss_t.index >= "2020-01-15") & (loss_t.index <= "2020-06-30")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(loss_t.index[zoom], loss_t[zoom] * 100, width=0.8, color="#9ca3af", label="Daily loss (negative = gain)")
    for k in fc:
        f = fc[k][zoom]
        ax.plot(f.index, f["VaR0.99"] * 100, lw=1.4, color=COLORS[k], label=NAMES[k])
        hit = loss_t[zoom] > f["VaR0.99"]
        ax.scatter(loss_t[zoom][hit].index + SHIFT[k], loss_t[zoom][hit] * 100, marker=MARKERS[k], s=40,
                   facecolors="none", edgecolors=COLORS[k], linewidths=1.2, zorder=3)
    ax.axhline(0, color="#6b7280", lw=0.6)
    ax.set_ylabel("One-day loss, %"); ax.legend(frameon=False, loc="upper right")
    ax.set_title("COVID crash: 99% VaR forecasts and breaches (dots)", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "charts", "var_covid.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    counts = {k: traffic_light(loss_t > fc[k]["VaR0.99"])[3] for k in fc}
    top = max(16, max(c.max() for c in counts.values()) + 2)
    ax.set_ylim(-0.5, top)
    ax.axhspan(10, top, color="#fee2e2", zorder=0)
    ax.axhspan(5, 10, color="#fef3c7", zorder=0)
    for k in fc:
        ax.plot(counts[k], lw=1.1, color=COLORS[k], label=NAMES[k])
    ax.set_ylabel("Breaches in last 250 days"); ax.legend(frameon=False, loc="upper right")
    ax.set_title("Rolling 250-day breach count at 99% (yellow and red: Basel zones)", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "charts", "breach_count.png"), dpi=150); plt.close(fig)

    for k in fc:
        e = ev[k]
        print(f"{k:6s} breaches {e['breaches']:3d}/{e['expected']:.0f}  kupiec {e['kupiec_p']:.3f}  ind {e['ind_p']:.3f}  dq {e['dq_p']:.3f}  "
              f"red {e['red']:.1%}  Z2 {e['z2']:.2f}  avgVaR {e['avg_var']:.2%}")
    decompose.main()
    print("README.md, results/ and charts/ updated")


if __name__ == "__main__":
    main()
