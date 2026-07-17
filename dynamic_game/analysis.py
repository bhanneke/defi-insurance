"""Validation and experiment analysis for the dynamic game."""
import numpy as np
import pandas as pd

from params import AllParams
from attacker import attack_cost
from protocols import precompute_attacker_matrix, best_response, payoff_of
from mechanism import coverage, utilization, U_max_dynamic, gamma_raw, \
    gamma_blended, term_structure_from_hazard, p_risk_anchor, \
    quarterly_pool_revenue
from protocols import xi_of_h


def incidents_frame(runs):
    rows = []
    for r in runs:
        for inc in r["incidents"]:
            rows.append(dict(inc, seed=r["seed"]))
    return pd.DataFrame(rows)


def epochs_frame(runs):
    rows = []
    for r in runs:
        for ep in r["epochs"]:
            rows.append(dict(ep, seed=r["seed"]))
    return pd.DataFrame(rows)


def frontier_regression(inc: pd.DataFrame, tau: float = 0.10):
    """Replicate the FIRM two-margin frontier estimation on SIMULATED data.

    Quantile regression at the lower envelope:
      log b = a + kappa*log e + (eta-1)*log V   (relative margin)
      log L = a + kappa*log e +  eta   *log V   (absolute margin)
    Should recover the calibrated eta (internal-consistency check: does the
    frontier estimator read back the primitive that generated the data?).
    """
    import statsmodels.formula.api as smf
    d = inc.copy()
    d = d[(d.L > 0) & (d.e > 0) & (d.V > 0)]
    d["logb"] = np.log(d.b_realized)
    d["logL"] = np.log(d.L)
    d["loge"] = np.log(d.e)
    d["logV"] = np.log(d.V)
    out = {}
    for margin, yvar in [("relative", "logb"), ("absolute", "logL")]:
        m = smf.quantreg(f"{yvar} ~ loge + logV", d).fit(q=tau)
        out[margin] = dict(
            const=float(m.params["Intercept"]),
            kappa_hat=float(m.params["loge"]),
            v_elast=float(m.params["logV"]),
            se_loge=float(m.bse["loge"]), se_logV=float(m.bse["logV"]),
            n=int(m.nobs))
    out["eta_hat_relative"] = out["relative"]["v_elast"] + 1.0
    out["eta_hat_absolute"] = out["absolute"]["v_elast"]
    return out


def floor_check(inc: pd.DataFrame, P: AllParams):
    """Prop. 2 (FIRM): every attack satisfies expected-b >= c(e*,V)/V."""
    c = attack_cost(inc.e.values, inc.V.values, P.attacker)
    floor = c / inc.V.values
    ok_expected = inc.b_expected.values * P.attacker.eps_mean >= floor - 1e-9
    # participation is in EXPECTED terms: E[eps]*b*V > c  <=>  E[eps]*b >= floor
    return dict(frac_above_floor=float(np.mean(ok_expected)),
                min_slack=float(np.min(P.attacker.eps_mean
                                       * inc.b_expected.values - floor)))


def deviation_check(P: AllParams, run, rng, n_check: int = 25):
    """Verify stage-Nash: no sampled unilateral (h, C_C) deviation improves
    the final epoch's payoff. Rebuilds the final stage and compares the
    equilibrium utility to the full grid maximum for sampled protocols.
    """
    pop = run["pop"]
    CLP = run["final_CLP"]
    from game import solve_stage
    eq = solve_stage(pop, CLP, P, state0=None)
    br, atk = eq["br"], eq["atk"]
    sum_CC = br["CC"].sum()
    C_total = CLP + sum_CC
    base, fees = quarterly_pool_revenue(C_total, float(br["p_q"].sum()), P.mech)
    aggregates = dict(gamma=eq["gamma"], gross_rev=base + fees,
                      sum_CC_others=sum_CC - br["CC"],
                      cap_scale=eq["cap_scale"])
    # Equilibrium payoff vs. grid max under FIXED aggregates
    br2 = best_response(pop, atk, aggregates, P)
    idx = rng.choice(pop["TVL"].shape[0], size=n_check, replace=False)
    u_eq = payoff_of(pop, atk, aggregates, P, br2["ih"], br2["CC"])
    gains = []
    hg = atk["h_grid"]
    for i in idx:
        best_dev = -np.inf
        for ihh in range(hg.size):
            ccs = np.linspace(0, P.proto.cc_tvl_cap * pop["TVL"][i], 40)
            for cc in ccs:
                u = payoff_of(
                    {k: v[i:i + 1] for k, v in pop.items()},
                    {k: (v[i:i + 1] if isinstance(v, np.ndarray) and v.ndim == 2
                         else v) for k, v in atk.items()},
                    dict(aggregates,
                         sum_CC_others=aggregates["sum_CC_others"][i:i + 1]),
                    P, np.array([ihh]), np.array([cc]))[0]
                best_dev = max(best_dev, u)
        gains.append(best_dev - u_eq[i])
    gains = np.array(gains)
    return dict(max_gain=float(gains.max()),
                mean_gain=float(gains.mean()),
                rel_max_gain=float((gains / (np.abs(u_eq[idx]) + 1e-9)).max()))


def summarize(runs, P: AllParams):
    ep = epochs_frame(runs)
    inc = incidents_frame(runs)
    n_years = P.game.n_quarters / 4.0
    per_run = ep.groupby("seed").agg(
        hacks=("n_hacks", "sum"), claims=("claims", "sum"),
        paid=("paid", "sum"), forfeited=("forfeited", "sum"),
        shortfall=("shortfall", "sum"), cov_mean=("coverage", "mean"),
        U=("U", "mean"), gamma=("gamma", "mean"), mean_h=("mean_h", "mean"),
        insured=("insured_frac", "mean"))
    cov_dollar_years = per_run.cov_mean * n_years
    loss_rate_bps = 1e4 * per_run.claims / cov_dollar_years
    loss_rate = float(loss_rate_bps.mean())
    raw_losses = inc.groupby("seed").L.sum() if len(inc) else pd.Series(dtype=float)
    out = dict(
        n_runs=len(runs),
        hacks_per_year=float(per_run.hacks.mean() / n_years),
        raw_losses_per_year=float(raw_losses.reindex(per_run.index)
                                  .fillna(0.0).mean() / n_years),
        mean_loss_per_hack=float(inc.L.mean()) if len(inc) else 0.0,
        median_loss=float(inc.L.median()) if len(inc) else 0.0,
        mean_blast=float(inc.b_realized.mean()) if len(inc) else 0.0,
        mean_effort=float(inc.e.mean()) if len(inc) else 0.0,
        loss_rate_bps=None if not np.isfinite(loss_rate) else loss_rate,
        mean_U=float(per_run.U.mean()), mean_gamma=float(per_run.gamma.mean()),
        mean_h=float(per_run.mean_h.mean()),
        insured_frac=float(per_run.insured.mean()),
        shortfall_runs=float((per_run.shortfall > 0).mean()),
        pred_vs_realized_hacks=(float(ep.groupby("seed").exp_hacks.sum().mean()),
                                float(per_run.hacks.mean())),
        fp_converged=float(ep.fp_conv.mean()),
    )
    return out, ep, inc
