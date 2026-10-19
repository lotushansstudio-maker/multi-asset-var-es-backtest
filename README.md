# Multi-Asset Volatility Forecasting and VaR/ES Backtest

How well do standard risk models forecast the next day's loss on a simple multi-asset portfolio, and where do they fail?

The portfolio holds US equities, US Treasuries, gold and the euro (through ETFs). Three models forecast one-day **Value at Risk (VaR)** and **Expected Shortfall (ES)** every day from 2008 onwards, using only data available the evening before. The forecasts are then backtested with the tests a bank risk team or regulator would use.

## Models

| Model | How the forecast is made |
|---|---|
| **Historical simulation** | The 99% VaR is the 99th percentile of the last 500 daily losses; ES is the average of the losses beyond it. No distribution is assumed, but it reacts slowly to a new crisis. |
| **EWMA (RiskMetrics)** | Volatility is an exponentially weighted average of past squared returns (lambda 0.94), with normally distributed returns. Reacts fast, but normal tails are thin. |
| **GARCH(1,1), Student-t** | Volatility follows sigma²ₜ = ω + α r²ₜ₋₁ + β sigma²ₜ₋₁, with fat-tailed Student-t shocks. Parameters are fitted by maximum likelihood on all data up to the forecast date and refitted every 20 trading days. |

Losses are positive numbers, so a 99% VaR of 1.2% means "a one-day loss larger than 1.2% should happen on about 1 day in 100". The 97.5% ES is the risk measure used under the Basel FRTB rules; 99% VaR is the classic one.

## Backtests

- **Kupiec test**: is the share of days with a loss above VaR (a *breach*) equal to 1%?
- **Christoffersen test**: are breaches independent, or does one breach make another more likely the next day? Clustered breaches mean the model is slow to react.
- **Conditional coverage**: both of the above together.
- **Basel traffic light**: breaches in each rolling 250-day window; 10 or more is the red zone.
- **Acerbi-Szekely Z2 test for ES**: are losses beyond VaR, on average, as large as the ES forecast said? 0 means right; above about 0.70 rejects at the 5% level.

## Key findings

- **None of the three models passes the 99% VaR backtest over 2008 to 2026.** Breaches run 31% to 82% above the 47 expected, and they cluster in crises (Christoffersen independence rejected for all three). The portfolio's daily loss tail is fatter, and arrives more suddenly, than any of the models allows.
- **GARCH(1,1) with Student-t errors comes closest:** 65 breaches, the fewest back-to-back breaches (4), the Basel red zone in only 1.7% of 250-day windows, and the best Expected Shortfall test (Z2 of 0.19, not rejected).
- **EWMA with normal returns is worst at 99% but fine at 95%** (239 breaches against 236 expected). The normal distribution describes ordinary days well and the tail badly.
- **Historical simulation reacts too slowly.** It had 15 breaches against 1.5 expected in the 2008 to 2009 crisis, and on the worst day of the test (18 March 2020, a 4.57% loss) its 99% VaR was 1.26%, against about 3.6% for EWMA and GARCH.
- **2022 hurt every model** (5 to 7 breaches against 2.5 expected), when stocks and bonds fell together. This is consistent with the limitation that the models work on the portfolio return and do not track the stock-bond correlation directly.
- **Expected Shortfall is not rejected for any model** (all Z2 values below 0.70), so the size of losses beyond VaR was roughly right even where the frequency of breaches was not.

## Results

<!-- RESULTS -->
Test period: **2007-12-31 to 2026-10-16**, 4,729 trading days. Portfolio: 40% SPY (US equities), 40% TLT (US Treasuries 20y+), 10% GLD (Gold), 10% FXE (Euro vs US dollar), rebalanced daily. Expected 99% VaR breaches: **47**.

| | Historical simulation (500 days) | EWMA, lambda 0.94 (normal) | GARCH(1,1), Student-t |
|---|---|---|---|
| 99% VaR breaches (expected 47) | 62 (1.31%) | 86 (1.82%) | 65 (1.37%) |
| Kupiec p-value (breach rate = 1%?) | 0.040 | <0.001 | 0.014 |
| Christoffersen independence p-value | <0.001 | <0.001 | 0.014 |
| Conditional coverage p-value | <0.001 | <0.001 | 0.002 |
| Back-to-back breaches | 8 | 7 | 4 |
| Share of 250-day windows in Basel red zone (10+ breaches) | 4.3% | 3.1% | 1.7% |
| 97.5% ES test, Acerbi-Szekely Z2 (0 = right; >0.70 rejects) | 0.26 | 0.45 | 0.19 |
| Average 99% VaR (one-day loss) | 1.39% | 1.21% | 1.30% |
| Average 97.5% ES | 1.47% | 1.22% | 1.33% |
| Verdict on 99% VaR | too many breaches | too many breaches | too many breaches |

Latest GARCH fit (2026-10-06): alpha 0.071, beta 0.917, persistence 0.987, Student-t degrees of freedom 8.7.
<!-- /RESULTS -->

![VaR forecasts and losses](charts/var_full.png)

### Stress periods: 99% VaR breaches

<!-- STRESS -->
| Period | Days | Expected | Historical simulation (500 days) | EWMA, lambda 0.94 (normal) | GARCH(1,1), Student-t |
|---|---|---|---|---|---|
| Global financial crisis (Sep 2008 to Mar 2009) | 146 | 1.5 | 15 | 6 | 7 |
| COVID crash (20 Feb to 30 Apr 2020) | 50 | 0.5 | 7 | 7 | 6 |
| Rates shock (2022) | 251 | 2.5 | 7 | 7 | 5 |

Worst day in the test: **2020-03-18**, a loss of 4.57%. Forecast 99% VaR that morning: historical simulation 1.26%, EWMA 3.55%, GARCH 3.60%.
<!-- /STRESS -->

![COVID crash](charts/var_covid.png)

![Rolling breach count](charts/breach_count.png)

### Robustness checks

<!-- ROBUST -->
| Variant | 99% VaR breaches | Kupiec p | Independence p | Z2 |
|---|---|---|---|---|
| Historical simulation, 250 days | 74 of 47 | <0.001 | <0.001 | 0.41 |
| Historical simulation, 1,000 days (from 2009-12-23) | 45 of 42 | 0.678 | <0.001 | 0.01 |
| EWMA, lambda 0.97 | 82 of 47 | <0.001 | <0.001 | 0.45 |
| Historical simulation (500 days), 95% VaR | 255 of 236 | 0.221 | <0.001 | |
| EWMA, lambda 0.94 (normal), 95% VaR | 239 of 236 | 0.865 | 0.027 | |
| GARCH(1,1), Student-t, 95% VaR | 246 of 236 | 0.527 | 0.006 | |
<!-- /ROBUST -->

## Limitations

- **One portfolio, fixed weights.** Daily rebalancing to fixed weights ignores trading costs and is a simplification; real portfolios drift and trade.
- **ETF prices, US hours.** All four ETFs trade in New York, so the test does not deal with asynchronous closing times across markets.
- **Portfolio-level models.** Each model is applied to the portfolio's return series directly. A full multivariate model (for example DCC-GARCH) would capture changing correlations between stocks and bonds, which matters in a year like 2022.
- **Free data.** Adjusted prices come from Yahoo Finance and can contain errors.
- **Backtests have low power.** With 1% VaR and a few thousand days, the tests can only reject models that are clearly wrong.

## Run it

```bash
pip install -r requirements.txt
python download.py   # daily adjusted prices from Yahoo Finance into data/
python backtest.py   # forecasts, backtests -> results/, charts/ and the sections above
```

## Files

| File | What it does |
|---|---|
| `download.py` | Downloads ETF prices with `yfinance` |
| `common.py` | Portfolio definition and daily returns |
| `models.py` | Historical simulation, EWMA and GARCH(1,1)-t forecasts of VaR and ES |
| `backtest.py` | Kupiec, Christoffersen, traffic light and Z2 tests; stress periods; robustness; charts |
| `results/` | Forecasts, fitted GARCH parameters and backtest statistics as CSV |
