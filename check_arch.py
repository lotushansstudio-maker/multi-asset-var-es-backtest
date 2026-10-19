"""Check our GARCH(1,1)-t and GJR-GARCH(1,1)-t fits against the `arch` package on the full sample.

Run after download.py:   python check_arch.py
"""
import numpy as np
from arch import arch_model

import models
from common import portfolio_returns, write_readme


def main():
    r, _ = portfolio_returns()
    x = r.values * 100
    md = ["| Model | Fit | omega | alpha | gamma | beta | nu | Log-likelihood |", "|---|---|---|---|---|---|---|---|"]
    for name, asym in (("GARCH(1,1)-t", False), ("GJR-GARCH(1,1)-t", True)):
        p, (omega, alpha, gamma, beta, nu), _ = models.fit_garch(x, asym=asym)
        ours = [omega, alpha, gamma, beta, nu, -models._neg_loglik(p, x)]
        res = arch_model(x, mean="Zero", vol="GARCH", p=1, o=int(asym), q=1, dist="t").fit(disp="off")
        q = res.params
        theirs = [q["omega"], q["alpha[1]"], q["gamma[1]"] if asym else 0.0, q["beta[1]"], q["nu"], res.loglikelihood]
        for label, v in (("this repo", ours), ("arch package", theirs)):
            md.append(f"| {name} | {label} | " + " | ".join(f"{u:.4f}" for u in v[:5]) + f" | {v[5]:,.1f} |")
        print(name, "ours", np.round(ours, 4), "arch", np.round(theirs, 4))
    md += ["", f"Full sample of {len(x):,} daily portfolio returns in percent, zero mean. The two fits start the variance "
           "recursion differently (this repo: sample variance; `arch`: an exponentially weighted backcast), so small "
           "differences in parameters and log-likelihood are expected."]
    write_readme("VALID", "\n".join(md))


if __name__ == "__main__":
    main()
