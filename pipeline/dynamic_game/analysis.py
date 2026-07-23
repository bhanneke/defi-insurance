"""Validation and experiment analysis for the dynamic game."""
import numpy as np
import pandas as pd

from params import AllParams
from attacker import attack_cost
from protocols import best_response, utility_cc, eps_quadrature
from mechanism import quarterly_pool_revenue


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


def deviation_check(P: AllParams, run, rng, n_check: int = 25,
                    n_dense: int = 4001):
    """Verify stage-Nash: no sampled unilateral (h, C_C) deviation improves
    the final epoch's payoff. Independent of the golden-section optimizer:
    scans a DENSE linear C_C grid (n_dense points) at every h level and
    compares against the equilibrium utility under fixed aggregates.
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
    br2 = best_response(pop, atk, aggregates, P)
    u_eq = br2["utility"]
    eps_n, eps_w = eps_quadrature(P)

    idx = rng.choice(pop["TVL"].shape[0], size=n_check, replace=False)
    gains = []
    for i in idx:
        cap = P.proto.cc_tvl_cap * pop["TVL"][i]
        ccs = np.linspace(0.0, cap, n_dense) if P.mech.insurance_on \
            else np.zeros(1)
        # replicate protocol i across the dense CC grid: shape (n_dense, n_h)
        pop_i = {k: np.repeat(v[i:i + 1], ccs.size) for k, v in pop.items()}
        atk_i = {k: (np.repeat(v[i:i + 1, :], ccs.size, axis=0)
                     if isinstance(v, np.ndarray) and v.ndim == 2 else v)
                 for k, v in atk.items()}
        agg_i = dict(aggregates,
                     sum_CC_others=np.repeat(
                         aggregates["sum_CC_others"][i:i + 1], ccs.size))
        CC_mat = np.broadcast_to(ccs[:, None],
                                 (ccs.size, atk["h_grid"].size)).copy()
        u_dev = utility_cc(CC_mat, pop_i, atk_i, agg_i, P, eps_n, eps_w)
        gains.append(float(u_dev.max()) - u_eq[i])
    gains = np.array(gains)
    return dict(max_gain=float(gains.max()),
                mean_gain=float(gains.mean()),
                rel_max_gain=float((gains / (np.abs(u_eq[idx]) + 1e-9)).max()))


def full_game_deviation(P: AllParams, run, top_k: int = 8):
    """Max unilateral gain when the deviator INTERNALIZES its effect on the
    mechanism aggregates (gamma, gross_rev, cap_scale, hazard index).

    The stage solver's certificate is for the aggregate-taking game; this
    measures the distance of the computed profile from FULL-game Nash, for
    the protocols with the largest collateral (largest price impact).
    Candidate deviations: dense local grid around the accepted C_C plus a
    coarse global grid, at every h level, with all aggregates recomputed
    per candidate.
    """
    from game import solve_stage, _aggregates_of
    pop, CLP = run["pop"], run["final_CLP"]
    eq = solve_stage(pop, CLP, P)
    br, atk = eq["br"], eq["atk"]
    hg = atk["h_grid"]
    n = pop["TVL"].shape[0]
    eps_n, eps_w = eps_quadrature(P)

    agg0, _ = _aggregates_of(br["CC"], br["h"], br["p_q"], CLP, pop, P)
    CC_mat = np.repeat(br["CC"][:, None], hg.size, axis=1)
    u0_all = utility_cc(CC_mat, pop, atk, agg0, P, eps_n, eps_w)
    u0 = u0_all[np.arange(n), br["ih"]]

    gains = {}
    for i in np.argsort(-br["CC"])[:top_k]:
        cap = P.proto.cc_tvl_cap * pop["TVL"][i]
        local = br["CC"][i] + np.linspace(-5.0, 5.0, 101)
        ccs = np.unique(np.clip(np.concatenate(
            [local, np.linspace(0.0, cap, 41)]), 0.0, cap))
        pop_i = {k: v[i:i + 1] for k, v in pop.items()}
        best = -np.inf
        for ih in range(hg.size):
            atk_i = {k: (v[i:i + 1, ih:ih + 1]
                         if isinstance(v, np.ndarray) and v.ndim == 2 else v)
                     for k, v in atk.items()}
            for c in ccs:
                CCd = br["CC"].copy(); CCd[i] = c
                hd = br["h"].copy(); hd[i] = hg[ih]
                pqd = br["p_q"].copy(); pqd[i] = atk["p_q"][i, ih]
                aggd, _ = _aggregates_of(CCd, hd, pqd, CLP, pop, P)
                agg_i = dict(aggd,
                             sum_CC_others=aggd["sum_CC_others"][i:i + 1])
                atk_h = dict(atk_i, h_grid=hg[ih:ih + 1])
                u = utility_cc(np.array([[c]]), pop_i, atk_h, agg_i, P,
                               eps_n, eps_w)[0, 0]
                best = max(best, u)
        gains[int(i)] = float(best - u0[i])
    arr = np.array(list(gains.values()))
    return dict(max_gain=float(arr.max()), mean_gain=float(arr.mean()),
                per_protocol=gains)


def summarize(runs, P: AllParams):
    ep = epochs_frame(runs)
    inc = incidents_frame(runs)
    n_years = P.game.n_quarters / 4.0
    per_run = ep.groupby("seed").agg(
        hacks=("n_hacks", "sum"), claims=("claims", "sum"),
        paid=("paid", "sum"), forfeited=("forfeited", "sum"),
        shortfall=("shortfall", "sum"), cov_mean=("coverage", "mean"),
        U=("U", "mean"), gamma=("gamma", "mean"), mean_h=("mean_h", "mean"),
        insured=("insured_frac", "mean"), sum_CC=("sum_CC", "mean"),
        CLP=("CLP", "mean"), r_pool=("r_pool_eff", "mean"))
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
        collateral_to_coverage=float((per_run.sum_CC / per_run.cov_mean).mean())
            if per_run.cov_mean.gt(0).all() else None,
        mean_C_total=float((per_run.sum_CC + per_run.CLP).mean()),
        final_r_pool=float(ep[ep.t == ep.t.max()].r_pool_eff.mean()),
        shortfall_runs=float((per_run.shortfall > 0).mean()),
        pred_vs_realized_hacks=(float(ep.groupby("seed").exp_hacks.sum().mean()),
                                float(per_run.hacks.mean())),
        fp_converged=float(ep.fp_conv.mean()),
    )
    return out, ep, inc
