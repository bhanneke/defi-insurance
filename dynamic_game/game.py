"""Dynamic game engine.

Solution concept: within each quarter, a stage Nash equilibrium computed by
fixed-point iteration over
  (i)   protocol best responses (C_C_i, h_i),
  (ii)  the attacker's best response (e*, participation) to (V_i, h_i),
  (iii) rational-expectations speculator pricing: HACK term structures equal
        the attack probabilities the attacker's strategy actually induces
        (Prop. 1 of the insurance paper, with the truth now endogenous),
  (iv)  the mechanism aggregates (U, U_max cap, gamma).
Across quarters, LP capital follows the Prop. 2(ii) adjustment dynamics and
TVL evolves with growth and realized hack losses (a Markov game with myopic
per-quarter payoffs; see README for the honest scope statement).
"""
import numpy as np

from params import AllParams
from attacker import attacker_best_response
from mechanism import (coverage, utilization, term_structure_from_hazard,
                       p_risk_anchor, U_max_dynamic, gamma_raw, gamma_blended,
                       quarterly_pool_revenue, hazard_from_term_structure)
from protocols import (init_population, precompute_attacker_matrix,
                       best_response, xi_of_h)


def solve_stage(pop, CLP, P: AllParams, state0=None):
    """Within-quarter fixed point. Returns the stage equilibrium."""
    mp, gp = P.mech, P.game
    n = pop["TVL"].shape[0]
    atk = precompute_attacker_matrix(pop["TVL"], P)

    # Initial iterate: previous epoch's strategy profile if available
    if state0 is not None:
        CC = state0["CC"].copy()
        p_q = state0["p_q"].copy()
    else:
        CC = 0.02 * pop["TVL"] if mp.insurance_on else np.zeros(n)
        p_q = np.full(n, 0.005)

    gamma, cap_scale, g_fair = 0.5, 1.0, 0.5
    info = dict(converged=False, iters=0)

    for it in range(gp.fp_max_iter):
        sum_CC = float(CC.sum())
        sum_CC_others = sum_CC - CC

        # Pricing layer: RE index from the current attack-probability iterate
        lam_bar = float(np.mean(-np.log1p(-np.clip(p_q, 0.0, 0.999)) / 0.25))
        P_ts = term_structure_from_hazard(lam_bar)
        lam_fit, p_anchor, p_risk = p_risk_anchor(P_ts, mp)

        # Mechanism aggregates
        xi = xi_of_h(state0["h"], P) if (it == 0 and state0 is not None) else None
        cov_now = coverage(CC, xi if xi is not None else P.proto.xi0, mp,
                           tvl=pop["TVL"])
        U = utilization(float(cov_now.sum()), CLP)
        U_cap = U_max_dynamic(p_anchor, mp)
        cap_scale = min(1.0, U_cap / max(U, 1e-9))
        g_raw = gamma_raw(min(U, U_cap), p_risk, p_anchor, mp)
        gamma, g_fair = gamma_blended(g_raw, CLP, sum_CC, mp)

        C_total = CLP + sum_CC
        exp_hacks = float(p_q.sum())
        base, fees = quarterly_pool_revenue(C_total, exp_hacks, mp)
        aggregates = dict(gamma=gamma, gross_rev=base + fees,
                          sum_CC_others=sum_CC_others, cap_scale=cap_scale)

        br = best_response(pop, atk, aggregates, P)

        d = gp.fp_damping
        p_new = (1 - d) * p_q + d * br["p_q"]
        delta = max(float(np.max(np.abs(p_new - p_q))),
                    float(np.max(np.abs(br["CC"] - CC))) / max(1.0, sum_CC))
        CC, p_q = br["CC"], p_new
        state0 = br  # carries h for the next iterate's xi
        info["iters"] = it + 1
        if delta < gp.fp_tol:
            info["converged"] = True
            break

    # Recompute consistent aggregates at the fixed point
    br = state0
    xi = xi_of_h(br["h"], P)
    cov = cap_scale * coverage(br["CC"], xi, mp, tvl=pop["TVL"])
    sum_CC = float(br["CC"].sum())
    lam_bar = float(np.mean(br["lam_ann"])) if "lam_ann" in br else \
        float(np.mean(-np.log1p(-np.clip(br["p_q"], 0.0, 0.999)) / 0.25))
    P_ts = term_structure_from_hazard(lam_bar)
    _, p_anchor, p_risk = p_risk_anchor(P_ts, mp)
    U = utilization(float(cov.sum()), CLP)
    U_cap = U_max_dynamic(p_anchor, mp)
    g_raw = gamma_raw(min(U, U_cap), p_risk, p_anchor, mp)
    gamma, g_fair = gamma_blended(g_raw, CLP, sum_CC, mp)

    return dict(br=br, atk=atk, cov=cov, U=U, U_cap=U_cap, cap_scale=cap_scale,
                gamma=gamma, gamma_fair=g_fair, gamma_raw=g_raw,
                p_anchor=p_anchor, p_risk=p_risk, lam_bar=lam_bar,
                sum_CC=sum_CC, info=info)


def run_game(P: AllParams, seed: int):
    """Simulate the dynamic game. Returns epoch panel + incident records."""
    mp, gp, ap = P.mech, P.game, P.attacker
    rng = np.random.default_rng(seed)
    pop = init_population(rng, P)
    CLP = gp.CLP0

    epochs, incidents = [], []
    stage_state = None
    lp_income_cum = prot_income_cum = op_income_cum = 0.0
    shortfall_cum = 0.0

    for t in range(gp.n_quarters):
        eq = solve_stage(pop, CLP, P, state0=stage_state)
        br = eq["br"]
        stage_state = br

        # --- Realize hacks: opportunity + execution are inside p_q ---
        hit = rng.random(pop["TVL"].shape[0]) < br["p_q"]
        eps = rng.beta(ap.eps_a, ap.eps_b, size=hit.sum())
        L = eps * br["b_star"][hit] * pop["TVL"][hit]
        cov_hit = eq["cov"][hit]
        claims = np.minimum(cov_hit, L)

        # --- Settlement (paper Sec. 4: forfeiture accrues to the pool and
        #     does NOT reduce LP payout obligations) ---
        total_claims = float(claims.sum())
        pay_total = min(total_claims, max(CLP, 0.0))
        shortfall = total_claims - pay_total
        CLP -= pay_total
        forfeited = float(br["CC"][hit].sum()) if (mp.forfeiture and mp.insurance_on) else 0.0
        CLP += forfeited

        # --- Yield accrual and split ---
        C_total = CLP + eq["sum_CC"]
        base, fees = quarterly_pool_revenue(C_total, int(hit.sum()), mp)
        gross = base + fees
        op_take = mp.phi * gross
        lp_take = eq["gamma"] * (1 - mp.phi) * gross
        prot_take = (1 - eq["gamma"]) * (1 - mp.phi) * gross
        CLP += lp_take

        lp_income_cum += lp_take - pay_total + forfeited
        prot_income_cum += prot_take - forfeited \
            - (mp.r_market / 4.0) * eq["sum_CC"]
        op_income_cum += op_take
        shortfall_cum += shortfall

        # --- LP capital adjustment, Prop. 2(ii) (expected-return based) ---
        exp_claims = float((br["p_q"] * np.minimum(
            eq["cov"], ap.eps_mean * br["b_star"] * pop["TVL"])).sum())
        r_LP_ann = 4.0 * (lp_take - exp_claims) / max(CLP, 1e-9)
        CLP *= (1.0 + mp.kappa_LP * (r_LP_ann - mp.r_market - mp.rho_LP) * 0.25)
        CLP = max(CLP, gp.CLP_floor)

        # --- Record ---
        epochs.append(dict(
            t=t, U=eq["U"], U_cap=eq["U_cap"], cap_scale=eq["cap_scale"],
            gamma=eq["gamma"], gamma_raw=eq["gamma_raw"],
            p_anchor=eq["p_anchor"], p_risk=eq["p_risk"], lam_bar=eq["lam_bar"],
            CLP=CLP, sum_CC=eq["sum_CC"], coverage=float(eq["cov"].sum()),
            n_hacks=int(hit.sum()), claims=total_claims, paid=pay_total,
            forfeited=forfeited, shortfall=shortfall,
            mean_h=float(br["h"].mean()), mean_p=float(br["p_q"].mean()),
            exp_hacks=float(br["p_q"].sum()),
            insured_frac=float((br["CC"] > 0).mean()),
            fp_iters=eq["info"]["iters"], fp_conv=eq["info"]["converged"],
        ))
        for j, idx in enumerate(np.where(hit)[0]):
            incidents.append(dict(
                t=t, V=pop["TVL"][idx], h=br["h"][idx], e=br["e_star"][idx],
                b_expected=br["b_star"][idx], eps=eps[j], L=L[j],
                b_realized=L[j] / pop["TVL"][idx], cov=eq["cov"][idx],
                claim=claims[j], CC=br["CC"][idx], pi_star=br["pi_star"][idx],
            ))

        # --- State transition ---
        g_q = (1.0 + gp.tvl_growth) ** 0.25 - 1.0
        pop["TVL"] *= (1.0 + g_q)
        pop["TVL"][hit] = np.maximum(pop["TVL"][hit] - L, P.proto.tvl_clip[0])

    return dict(epochs=epochs, incidents=incidents, pop=pop,
                lp_income=lp_income_cum, prot_income=prot_income_cum,
                op_income=op_income_cum, shortfall=shortfall_cum,
                final_CLP=CLP, seed=seed,
                final_h=stage_state["h"].copy(),
                final_p_q=stage_state["p_q"].copy(),
                final_CC=stage_state["CC"].copy())


def run_mc(P: AllParams, n_seeds: int, base_seed: int = 1000):
    return [run_game(P, base_seed + s) for s in range(n_seeds)]
