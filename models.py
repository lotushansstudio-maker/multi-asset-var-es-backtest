"""One-day-ahead VaR and Expected Shortfall forecasts from three models.

Every forecast for day t uses only returns up to day t-1. Losses are positive numbers (loss = -return),
so a VaR of 0.012 means "a one-day loss of more than 1.2% should happen 1% of the time".
"""
import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

LEVELS = (0.95, 0.975, 0.99)


# ---------- 1. Historical simulation ----------
def historical(loss, window=500, levels=LEVELS):
    """Empirical quantile of the last `window` daily losses; ES is the average loss beyond that quantile."""
    out = {}
    for a in levels:
        var = loss.rolling(window).quantile(a, interpolation="linear")
        es = loss.rolling(window).apply(lambda x, a=a: x[x >= np.quantile(x, a)].mean(), raw=True)
        out[f"VaR{a}"], out[f"ES{a}"] = var.shift(1), es.shift(1)
    return pd.DataFrame(out)


# ---------- 2. EWMA (RiskMetrics) ----------
def ewma_sigma(ret, lam=0.94, seed=60):
    """sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}; started from the variance of the first `seed` days."""
    r2 = ret.values ** 2
    s2 = np.empty(len(r2))
    s2[0] = r2[:seed].mean()
    for t in range(1, len(r2)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r2[t - 1]
    return pd.Series(np.sqrt(s2), index=ret.index)


def normal_var_es(sigma, levels=LEVELS):
    out = {}
    for a in levels:
        z = stats.norm.ppf(a)
        out[f"VaR{a}"], out[f"ES{a}"] = sigma * z, sigma * stats.norm.pdf(z) / (1 - a)
    return pd.DataFrame(out)


def ewma(ret, lam=0.94, levels=LEVELS):
    return normal_var_es(ewma_sigma(ret, lam), levels)


# ---------- 3. GARCH(1,1) with Student-t errors ----------
def garch_filter(r, omega, alpha, beta, s0=None):
    """Conditional variance path: sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}.
    Returns len(r) + 1 values; the last one is the forecast for the day after the sample."""
    if s0 is None:
        s0 = r.var()   # day 0 starts at the sample variance
    x = np.concatenate([[s0], omega + alpha * r ** 2])
    return signal.lfilter([1.0], [1.0, -beta], x)


def _unpack(p):
    """Map unconstrained parameters to omega > 0, alpha, beta >= 0 with alpha + beta < 1, nu > 2."""
    omega = np.exp(p[0])
    persist = 1 / (1 + np.exp(-p[1]))          # alpha + beta, in (0, 1)
    share = 1 / (1 + np.exp(-p[2]))            # alpha / (alpha + beta)
    nu = 2.05 + np.exp(p[3])
    return omega, persist * share, persist * (1 - share), nu


def _neg_loglik(p, r):
    omega, alpha, beta, nu = _unpack(p)
    s2 = garch_filter(r, omega, alpha, beta)[:-1]
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    # Student-t rescaled to unit variance: r_t = sigma_t * sqrt((nu-2)/nu) * T_nu
    scale = np.sqrt(s2 * (nu - 2) / nu)
    return -np.sum(stats.t.logpdf(r / scale, nu) - np.log(scale))


def fit_garch(r, start=None):
    """Maximum-likelihood fit of GARCH(1,1)-t. `r` is in percent to keep the optimiser well scaled."""
    if start is None:
        v = r.var()
        start = np.array([np.log(0.05 * v), 2.9, 2.2, np.log(6.0)])   # alpha+beta ~0.95, alpha ~0.1
    res = optimize.minimize(_neg_loglik, start, args=(r,), method="L-BFGS-B")
    return res.x, _unpack(res.x)


def t_var_es(sigma, nu, a):
    """VaR and ES of sigma * (unit-variance Student-t) at confidence level a."""
    q = stats.t.ppf(a, nu)
    k = np.sqrt((nu - 2) / nu)
    es_std = stats.t.pdf(q, nu) / (1 - a) * (nu + q ** 2) / (nu - 1)
    return sigma * k * q, sigma * k * es_std


def garch(ret, start=500, refit=20, levels=LEVELS):
    """Refit every `refit` days on all data so far (expanding window), forecast the next days with the
    latest parameters. Returns forecasts and the parameter history."""
    r = ret.values * 100
    n = len(r)
    sig = np.full(n, np.nan)
    nus = np.full(n, np.nan)
    params, p0 = [], None
    s0 = r[:start].var()   # fixed starting variance, so no later data leaks into the path
    for i in range(start, n, refit):
        p0, (omega, alpha, beta, nu) = fit_garch(r[:i], p0)
        params.append({"date": ret.index[i], "omega": omega, "alpha": alpha, "beta": beta, "nu": nu})
        j = min(i + refit, n)
        s2 = garch_filter(r[:j], omega, alpha, beta, s0)   # s2[t] uses data up to t-1
        sig[i:j] = np.sqrt(s2[i:j])
        nus[i:j] = nu
    out = {}
    for a in levels:
        var, es = t_var_es(sig, nus, a)
        out[f"VaR{a}"], out[f"ES{a}"] = var / 100, es / 100
    return pd.DataFrame(out, index=ret.index), pd.DataFrame(params).set_index("date")
