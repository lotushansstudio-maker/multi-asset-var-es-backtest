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


# ---------- 3. GARCH(1,1) and GJR-GARCH(1,1) with Student-t errors ----------
def garch_filter(r, omega, alpha, beta, s0=None, gamma=0.0):
    """Conditional variance path: sigma^2_t = omega + (alpha + gamma * [r_{t-1} < 0]) * r^2_{t-1} + beta * sigma^2_{t-1}.
    gamma = 0 is plain GARCH; gamma > 0 (GJR) lets losses raise volatility more than gains.
    Returns len(r) + 1 values; the last one is the forecast for the day after the sample."""
    if s0 is None:
        s0 = r.var()   # day 0 starts at the sample variance
    x = np.concatenate([[s0], omega + (alpha + gamma * (r < 0)) * r ** 2])
    return signal.lfilter([1.0], [1.0, -beta], x)


def _unpack(p):
    """Map unconstrained parameters to omega > 0, alpha, gamma, beta >= 0 with persistence < 1, nu > 2.
    Four parameters give plain GARCH (gamma = 0); five give GJR, whose persistence is alpha + gamma / 2 + beta."""
    omega = np.exp(p[0])
    persist = 1 / (1 + np.exp(-p[1]))          # alpha + gamma / 2 + beta, in (0, 1)
    nu = 2.05 + np.exp(p[-1])
    if len(p) == 4:
        share = 1 / (1 + np.exp(-p[2]))        # alpha / (alpha + beta)
        return omega, persist * share, 0.0, persist * (1 - share), nu
    w = np.exp([p[2], p[3], 0.0])
    w = w / w.sum()                             # shares of alpha, gamma / 2 and beta
    return omega, persist * w[0], 2 * persist * w[1], persist * w[2], nu


def _neg_loglik(p, r):
    omega, alpha, gamma, beta, nu = _unpack(p)
    s2 = garch_filter(r, omega, alpha, beta, gamma=gamma)[:-1]
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    # Student-t rescaled to unit variance: r_t = sigma_t * sqrt((nu-2)/nu) * T_nu
    scale = np.sqrt(s2 * (nu - 2) / nu)
    return -np.sum(stats.t.logpdf(r / scale, nu) - np.log(scale))


def fit_garch(r, start=None, asym=False):
    """Maximum-likelihood fit of GARCH(1,1)-t or GJR-GARCH(1,1)-t. `r` is in percent to keep the optimiser well scaled."""
    if start is None:
        v = r.var()
        if asym:   # alpha ~0.04, gamma ~0.08, beta ~0.87
            start = np.array([np.log(0.05 * v), 2.9, -3.08, -3.08, np.log(6.0)])
        else:     # alpha+beta ~0.95, alpha ~0.1
            start = np.array([np.log(0.05 * v), 2.9, 2.2, np.log(6.0)])
    res = optimize.minimize(_neg_loglik, start, args=(r,), method="L-BFGS-B")
    return res.x, _unpack(res.x), res.fun


def refit_garch(r, prev=None, asym=False):
    """Fit from the previous refit's parameters and from the default start, keep the higher likelihood.
    Warm starts alone can get stuck: once a share parameter drifts far out (alpha near 0), its gradient vanishes."""
    fits = [fit_garch(r, None, asym)] + ([fit_garch(r, prev, asym)] if prev is not None else [])
    x, theta, _ = min(fits, key=lambda f: f[2])
    return x, theta


def t_var_es(sigma, nu, a):
    """VaR and ES of sigma * (unit-variance Student-t) at confidence level a."""
    q = stats.t.ppf(a, nu)
    k = np.sqrt((nu - 2) / nu)
    es_std = stats.t.pdf(q, nu) / (1 - a) * (nu + q ** 2) / (nu - 1)
    return sigma * k * q, sigma * k * es_std


def garch(ret, start=500, refit=20, levels=LEVELS, asym=False, fhs=False):
    """Refit every `refit` days on all data so far (expanding window), forecast the next days with the
    latest parameters. Returns forecasts and the parameter history.

    fhs=True gives filtered historical simulation: instead of the Student-t quantile, VaR and ES come from the
    empirical distribution of past standardised losses (loss / GARCH sigma), scaled by tomorrow's sigma. The pool
    of standardised losses is rebuilt at each refit from data before the refit date."""
    r = ret.values * 100
    n = len(r)
    sig = np.full(n, np.nan)
    nus = np.full(n, np.nan)
    zq = {a: np.full(n, np.nan) for a in levels}
    ze = {a: np.full(n, np.nan) for a in levels}
    params, p0 = [], None
    s0 = r[:start].var()   # fixed starting variance, so no later data leaks into the path
    for i in range(start, n, refit):
        p0, (omega, alpha, gamma, beta, nu) = refit_garch(r[:i], p0, asym)
        params.append({"date": ret.index[i], "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta, "nu": nu})
        j = min(i + refit, n)
        s2 = garch_filter(r[:j], omega, alpha, beta, s0, gamma)   # s2[t] uses data up to t-1
        sig[i:j] = np.sqrt(s2[i:j])
        nus[i:j] = nu
        if fhs:
            z = -r[:i] / np.sqrt(s2[:i])   # standardised losses up to day i-1
            for a in levels:
                q = np.quantile(z, a)
                zq[a][i:j], ze[a][i:j] = q, z[z >= q].mean()
    out = {}
    for a in levels:
        var, es = (sig * zq[a], sig * ze[a]) if fhs else t_var_es(sig, nus, a)
        out[f"VaR{a}"], out[f"ES{a}"] = var / 100, es / 100
    params = pd.DataFrame(params).set_index("date")
    if not asym:
        params = params.drop(columns="gamma")
    return pd.DataFrame(out, index=ret.index), params
