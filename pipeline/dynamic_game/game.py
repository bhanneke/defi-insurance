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
                       quarterly_pool_revenue, pool_return)
from protocols import (init_population, precompute_attacker_matrix,
                       best_response, xi_of_h, utility_cc, eps_quadrature)


def _aggregates_of(CC, h, p_q, CLP, pop, P: AllParams):
    """Self-consistent mechanism aggregates induced by a strategy profile.

    Used both inside the fixed-point iteration and for the epsilon-Nash
    certificate, so profile and aggregates always share one convention.
    """
    mp = P.mech
    sum_CC = float(CC.sum())
    lam_bar = float(np.mean(-np.log1p(-np.clip(p_q, 0.0, 0.999)) / 0.25))
    P_ts = term_structure_from_hazard(lam_bar)
    _, p_anchor, p_risk = p_risk_anchor(P_ts, mp)
    cov_nom = coverage(CC, xi_of_h(h, P), mp, tvl=pop["TVL"])
    U = utilization(float(cov_nom.sum()), CLP)
    U_cap = U_max_dynamic(p_anchor, mp)
    cap_scale = min(1.0, U_cap / max(U, 1e-9))
    g_raw = gamma_raw(min(U, U_cap), p_risk, p_anchor, mp)
    gamma, g_fair = gamma_blended(g_raw, CLP, sum_CC, mp)
    C_total = CLP + sum_CC
    base, fees = quarterly_pool_revenue(C_total, float(p_q.sum()), mp)
    agg = dict(gamma=gamma, gross_rev=base + fees,
               sum_CC_others=sum_CC - CC, cap_scale=cap_scale)
    extras = dict(U=U, U_cap=U_cap, cap_scale=cap_scale, gamma=gamma,
                  gamma_fair=g_fair, gamma_raw=g_raw, p_anchor=p_anchor,
                  p_risk=p_risk, lam_bar=lam_bar, sum_CC=sum_CC,
                  cov=cap_scale * cov_nom)
    return agg, extras


def _eps_of(br, agg, pop, atk, P: AllParams, eps_n, eps_w):
    """Certified epsilon: per-protocol unilateral (h, C_C) utility gains
    over the profile br when aggregates agg are held fixed (Nash criterion).
    Returns (max gain, per-protocol gains, best-response profile)."""
    n = pop["TVL"].shape[0]
    br_star = best_response(pop, atk, agg, P)
    CC_mat = np.repeat(br["CC"][:, None], atk["h_grid"].size, axis=1)
    u_all = utility_cc(CC_mat, pop, atk, agg, P, eps_n, eps_w)
    u_acc = u_all[np.arange(n), br["ih"]]
    gains = br_star["utility"] - u_acc
    return float(np.max(gains)), gains, br_star


def _br_single(i, agg, pop, atk, P: AllParams):
    """Best response of protocol i alone under fixed aggregates."""
    pop_i = {k: v[i:i + 1] for k, v in pop.items()}
    atk_i = {k: (v[i:i + 1] if isinstance(v, np.ndarray) and v.ndim == 2
                 else v) for k, v in atk.items()}
    agg_i = dict(agg, sum_CC_others=np.asarray(agg["sum_CC_others"])[i:i + 1])
    return best_response(pop_i, atk_i, agg_i, P)


def _sequential_refine(br, CLP, pop, atk, P: AllParams, eps_n, eps_w,
                       max_passes: int = 40):
    """Gauss-Seidel tail: sequentially re-optimize only the protocols with
    positive certified gain, refreshing aggregates after each update. This
    breaks the 2-cycles that simultaneous best response produces when a few
    protocols are near-indifferent between adjacent h-grid points."""
    gp = P.game
    prof = {k: br[k].copy() for k in ("CC", "h", "ih", "p_q", "b_star",
                                      "e_star", "pi_star", "xi", "utility")}
    best_prof, best_eps = None, np.inf
    for _ in range(max_passes):
        agg, _ = _aggregates_of(prof["CC"], prof["h"], prof["p_q"], CLP, pop, P)
        eps_k, gains, _ = _eps_of(prof, agg, pop, atk, P, eps_n, eps_w)
        if eps_k < best_eps:
            best_prof, best_eps = {k: v.copy() for k, v in prof.items()}, eps_k
        if eps_k < gp.fp_eps_tol:
            break
        for i in np.argsort(-gains):
            if gains[i] <= gp.fp_eps_tol:
                break
            agg, _ = _aggregates_of(prof["CC"], prof["h"], prof["p_q"],
                                    CLP, pop, P)
            bri = _br_single(int(i), agg, pop, atk, P)
            for k in prof:
                prof[k][i] = bri[k][0]
    return best_prof, best_eps


def solve_stage(pop, CLP, P: AllParams, state0=None):
    """Within-quarter stage equilibrium.

    Phase 1: damped fixed-point iteration for global stability.
    Phase 2: undamped best-response polish on self-consistent aggregates,
    keeping the profile with the smallest certified epsilon. The reported
    `eps_stage` is the max unilateral utility gain at the accepted profile
    — the equilibrium criterion itself. Exact pure-strategy equilibrium
    need not exist in the finite-h game (Nash 1951 guarantees only the
    mixed extension); the certificate quantifies the distance honestly.
    """
    mp, gp = P.mech, P.game
    n = pop["TVL"].shape[0]
    atk = precompute_attacker_matrix(pop["TVL"], P)
    eps_n, eps_w = eps_quadrature(P)

    # Initial iterate: previous epoch's strategy profile if available
    if state0 is not None:
        CC = state0["CC"].copy()
        h = state0["h"].copy()
        p_q = state0["p_q"].copy()
    else:
        CC = 0.02 * pop["TVL"] if mp.insurance_on else np.zeros(n)
        h = np.full(n, 0.0)
        p_q = np.full(n, 0.005)

    info = dict(converged=False, iters=0, polish_iters=0)
    br = None

    # --- Phase 1: damped iteration ---
    for it in range(gp.fp_max_iter):
        agg, _ = _aggregates_of(CC, h, p_q, CLP, pop, P)
        br = best_response(pop, atk, agg, P)
        d = gp.fp_damping
        p_new = (1 - d) * p_q + d * br["p_q"]
        CC_new = (1 - d) * CC + d * br["CC"]
        delta = max(float(np.max(np.abs(p_new - p_q))),
                    float(np.max(np.abs(CC_new - CC))) / max(1.0, CC.sum()))
        CC, p_q, h = CC_new, p_new, br["h"]
        info["iters"] = it + 1
        if delta < gp.fp_tol:
            break

    # --- Phase 2: undamped polish, keep the min-epsilon profile ---
    best_br, best_eps = None, np.inf
    for k in range(gp.polish_max_iter):
        agg, _ = _aggregates_of(br["CC"], br["h"], br["p_q"], CLP, pop, P)
        eps_k, _, br_star = _eps_of(br, agg, pop, atk, P, eps_n, eps_w)
        if eps_k < best_eps:
            best_br, best_eps = br, eps_k
        info["polish_iters"] = k + 1
        if eps_k < gp.fp_eps_tol:
            break
        br = br_star   # undamped best-response step

    # --- Phase 3: sequential (Gauss-Seidel) refinement if cycling ---
    if best_eps >= gp.fp_eps_tol:
        seq_prof, seq_eps = _sequential_refine(best_br, CLP, pop, atk, P,
                                               eps_n, eps_w)
        if seq_eps < best_eps:
            best_br, best_eps = seq_prof, seq_eps

    # --- Phase 4: exhaustive enumeration over a small flip set ---
    # If an exact pure equilibrium exists among the cycling candidates,
    # this finds it; if not, the residual epsilon is fundamental (finite
    # games need not admit exact pure-strategy Nash equilibria).
    if best_eps >= gp.fp_eps_tol:
        from itertools import product as _product
        agg, _ = _aggregates_of(best_br["CC"], best_br["h"], best_br["p_q"],
                                CLP, pop, P)
        _, gains, br_star = _eps_of(best_br, agg, pop, atk, P, eps_n, eps_w)
        flip = np.where(gains > gp.fp_eps_tol)[0]
        keys = ("CC", "h", "ih", "p_q", "b_star", "e_star", "pi_star",
                "xi", "utility")
        if 0 < flip.size <= 5:
            for combo in _product([0, 1], repeat=flip.size):
                if not any(combo):
                    continue
                prof = {k: best_br[k].copy() for k in keys}
                for b, i in zip(combo, flip):
                    if b:
                        for k in keys:
                            prof[k][i] = br_star[k][i]
                agg_c, _ = _aggregates_of(prof["CC"], prof["h"], prof["p_q"],
                                          CLP, pop, P)
                eps_c, _, _ = _eps_of(prof, agg_c, pop, atk, P, eps_n, eps_w)
                if eps_c < best_eps:
                    best_br, best_eps = prof, eps_c
                if best_eps < gp.fp_eps_tol:
                    break

    br = best_br
    info["eps_stage"] = best_eps
    info["converged"] = best_eps < gp.fp_eps_tol

    _, ex = _aggregates_of(br["CC"], br["h"], br["p_q"], CLP, pop, P)
    return dict(br=br, atk=atk, info=info, **ex)


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
        claims = (1.0 - mp.coinsurance) * np.minimum(cov_hit, L)

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
        exp_claims = (1.0 - mp.coinsurance) * float((br["p_q"] * np.minimum(
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
            r_pool_eff=pool_return(C_total, mp),
            n_hacks=int(hit.sum()), claims=total_claims, paid=pay_total,
            forfeited=forfeited, shortfall=shortfall,
            mean_h=float(br["h"].mean()), mean_p=float(br["p_q"].mean()),
            exp_hacks=float(br["p_q"].sum()),
            insured_frac=float((br["CC"] > 0).mean()),
            fp_iters=eq["info"]["iters"], fp_conv=eq["info"]["converged"],
            eps_stage=eq["info"].get("eps_stage", np.nan),
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
