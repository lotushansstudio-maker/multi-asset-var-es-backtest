import os, sys
import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
from backtest import kupiec, christoffersen
from decompose import component_var
from common import WEIGHTS


def test_kupiec_zero_when_rate_equals_p():
    hits = pd.Series([1] + [0] * 99)
    lr, p = kupiec(hits, 0.01)
    assert lr == pytest.approx(0, abs=1e-12) and p == pytest.approx(1)


def test_kupiec_hand_computed():
    # n = 10, x = 2, p = 0.1: LR = -2 * [8 ln 0.9 + 2 ln 0.1 - 8 ln 0.8 - 2 ln 0.2] = 0.888059
    lr, _ = kupiec(pd.Series([1, 1] + [0] * 8), 0.1)
    assert lr == pytest.approx(0.888059, abs=1e-5)


def test_christoffersen_counts_back_to_back():
    # transitions: 01 11 10 01 11 11 10 -> three 1-to-1 transitions
    _, _, n11 = christoffersen(pd.Series([0, 1, 1, 0, 1, 1, 1, 0]))
    assert n11 == 3


def test_t_var_es_matches_monte_carlo():
    nu, a = 5.0, 0.99
    rng = np.random.default_rng(0)
    x = rng.standard_t(nu, 4_000_000) * np.sqrt((nu - 2) / nu)   # unit variance
    q = np.quantile(x, a)
    var, es = models.t_var_es(1.0, nu, a)
    assert var == pytest.approx(q, rel=0.01)
    assert es == pytest.approx(x[x >= q].mean(), rel=0.01)


def test_garch_filter_matches_loop():
    rng = np.random.default_rng(1)
    r = rng.normal(size=300)
    omega, alpha, gamma, beta, s0 = 0.02, 0.05, 0.08, 0.9, 1.3
    s2 = [s0]
    for x in r:
        s2.append(omega + (alpha + gamma * (x < 0)) * x ** 2 + beta * s2[-1])
    np.testing.assert_allclose(models.garch_filter(r, omega, alpha, beta, s0, gamma), s2)


def test_gjr_with_zero_gamma_is_garch():
    rng = np.random.default_rng(2)
    r = rng.normal(size=300)
    np.testing.assert_allclose(models.garch_filter(r, 0.02, 0.08, 0.9, 1.0, 0.0), models.garch_filter(r, 0.02, 0.08, 0.9, 1.0))


@pytest.mark.parametrize("kw", [dict(asym=True, fhs=True), dict(asym=False, fhs=False)])
def test_forecasts_use_no_future_data(kw):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-01", periods=700)
    ret = pd.Series(rng.standard_t(5, 700) * 0.008, index=idx)
    base, _ = models.garch(ret, start=500, refit=20, **kw)
    shocked = ret.copy()
    shocked.iloc[650] = -0.2   # a huge loss on day 650
    alt, _ = models.garch(shocked, start=500, refit=20, **kw)
    # forecasts up to and including day 650 are made before its return is known
    pd.testing.assert_frame_equal(base.iloc[:651], alt.iloc[:651])
    assert not np.allclose(base.iloc[651:].values, alt.iloc[651:].values)


def test_component_var_adds_up_to_portfolio_ewma_var():
    rng = np.random.default_rng(4)
    rets = pd.DataFrame(rng.normal(scale=0.01, size=(400, 4)), columns=list(WEIGHTS.index))
    comp, share, _ = component_var(rets)
    port = models.ewma(rets @ WEIGHTS)["VaR0.99"]
    np.testing.assert_allclose(comp.sum(axis=1), port)
    np.testing.assert_allclose(share.sum(axis=1), 1.0)
