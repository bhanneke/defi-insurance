"""Protocol population and best responses.

Protocols now choose BOTH the collateral C_C (as in the insurance paper's
Eq. 7 objective) and a security level h (new margin), anticipating the
strategic attacker's response p_q(V, h) and b*(V, h) from the FIRM model.

Objective (quarterly, $M), extending Eq. (7) of the insurance paper:

  U(C_C, h) =  p * E[min(cov, L)]              expected insurance payout
             - 1{forfeiture} * p * C_C          collateral forfeited on hack
             + (1-p) * yield_share(C_C)         protocol share of pool yield
             - (r_market/4) * C_C               collateral opportunity cost
             - (c_h/4) * h * TVL                security expenditure
             - p * E[L]                         expected raw loss
             - rho_P * p * E[L - min(cov, L)]   risk-aversion penalty on the
                                                UNINSURED residual

The last term replaces the paper's "+rho_P p E[min(cov,L)]" utility bonus:
penalizing residual exposure is equivalent for the C_C margin up to a term
constant in C_C, and it prices the security margin h correctly.
L = eps * b*(V,h) * V with execution noise eps ~ Beta(eps_a, eps_b).
"""
import numpy as np
from scipy import stats

from params import AllParams
from attacker import attacker_best_response
from mechanism import coverage


def init_population(rng: np.random.Generator, P: AllParams):
    pp = P.proto
    n = pp.n_protocols
    TVL = pp.tvl_scale * (1.0 + rng.pareto(pp.tvl_pareto_a, size=n))
    TVL = np.clip(TVL, *pp.tvl_clip)
    rhoP = rng.lognormal(mean=np.log(pp.rho_med), sigma=pp.rho_sigma, size=n)
    rhoP = np.clip(rhoP, *pp.rho_clip)
    return dict(TVL=TVL, rhoP=rhoP)


def xi_of_h(h, P: AllParams):
    pp = P.proto
    return np.clip(pp.xi0 + pp.xi1 * h, 1e-6, pp.xi_cap)


def h_grid(P: AllParams):
    return np.linspace(0.0, P.proto.h_max, P.proto.n_h_grid)


def cc_grid_fractions(P: AllParams):
    """Collateral grid as fractions of the per-protocol cap (0 included)."""
    return np.concatenate([[0.0], np.logspace(-3, 0, P.proto.n_cc_grid - 1)])


def eps_quadrature(P: AllParams, K: int = 7):
    """Quantile-midpoint nodes for the Beta execution-noise expectation."""
    q = (np.arange(K) + 0.5) / K
    nodes = stats.beta.ppf(q, P.attacker.eps_a, P.attacker.eps_b)
    weights = np.full(K, 1.0 / K)
    return nodes, weights


def precompute_attacker_matrix(TVL, P: AllParams):
    """Attacker best response to every hypothetical security level.

    V is fixed within an epoch, so this is computed ONCE per epoch:
    arrays of shape (n_protocols, n_h) over the shared h grid.
    """
    hg = h_grid(P)
    Vm, Hm = np.meshgrid(TVL, hg, indexing="ij")
    br = attacker_best_response(Vm.ravel(), Hm.ravel(), P.attacker)
    shape = Vm.shape
    return {k: v.reshape(shape) for k, v in br.items()} | {"h_grid": hg}


def best_response(pop, atk, aggregates, P: AllParams):
    """Vectorized joint argmax over the (h, C_C) grid for every protocol.

    aggregates: dict with
      gamma         : blended LP yield share
      gross_rev     : quarterly pool revenue (base + fees), $M
      sum_CC_others : (n,) posted collateral of the OTHER protocols
      cap_scale     : prudential-cap coverage scaling, min(1, U_max/U)
    Returns dict with chosen h, CC and the induced attacker outcomes.
    """
    mp, pp, ap = P.mech, P.proto, P.attacker
    TVL, rhoP = pop["TVL"], pop["rhoP"]
    n = TVL.shape[0]
    hg = atk["h_grid"]

    if mp.insurance_on:
        fr = cc_grid_fractions(P)
        CC_grid = fr[None, :] * (pp.cc_tvl_cap * TVL)[:, None]      # (n, n_cc)
    else:
        CC_grid = np.zeros((n, 1))
    n_h, n_cc = hg.size, CC_grid.shape[1]

    eps_n, eps_w = eps_quadrature(P)

    p = atk["p_q"][:, :, None]                                       # (n, n_h, 1)
    b = atk["b_star"][:, :, None]
    E_loss = ap.eps_mean * b * TVL[:, None, None]                    # E[L]

    xi = xi_of_h(hg, P)                                              # (n_h,)
    cov = coverage(CC_grid[:, None, :], xi[None, :, None], mp,
                   tvl=TVL[:, None, None])                           # (n, n_h, n_cc)
    cov = aggregates["cap_scale"] * cov

    # E[min(cov, eps*b*V)] via quadrature over eps
    L_nodes = eps_n[None, None, None, :] * b[..., None] * TVL[:, None, None, None]
    payout = (np.minimum(cov[..., None], L_nodes) * eps_w).sum(axis=-1)
    residual = E_loss - payout

    # Pro-rata protocol yield share; own CC also enters the denominator.
    so = np.asarray(aggregates["sum_CC_others"])[:, None, None]
    w_yield = CC_grid[:, None, :] / np.maximum(so + CC_grid[:, None, :], 1e-9)
    ys = (1.0 - aggregates["gamma"]) * (1.0 - mp.phi) * aggregates["gross_rev"] * w_yield

    forfeit = (p * CC_grid[:, None, :]) if (mp.forfeiture and mp.insurance_on) else 0.0

    util = (p * payout
            - forfeit
            + (1.0 - p) * ys
            - (mp.r_market / 4.0) * CC_grid[:, None, :]
            - (pp.c_h / 4.0) * hg[None, :, None] * TVL[:, None, None]
            - p * E_loss
            - rhoP[:, None, None] * p * residual)

    flat = util.reshape(n, n_h * n_cc)
    best = np.argmax(flat, axis=1)
    ih, icc = np.unravel_index(best, (n_h, n_cc))
    rows = np.arange(n)

    return dict(
        h=hg[ih], CC=CC_grid[rows, icc], ih=ih,
        p_q=atk["p_q"][rows, ih], b_star=atk["b_star"][rows, ih],
        e_star=atk["e_star"][rows, ih], pi_star=atk["pi_star"][rows, ih],
        xi=xi_of_h(hg[ih], P), utility=flat[rows, best],
    )


def payoff_of(pop, atk, aggregates, P: AllParams, h_idx, CC_val):
    """Utility of an arbitrary (h index, C_C) profile — for deviation checks."""
    mp, pp, ap = P.mech, P.proto, P.attacker
    TVL, rhoP = pop["TVL"], pop["rhoP"]
    hg = atk["h_grid"]
    rows = np.arange(TVL.shape[0])
    eps_n, eps_w = eps_quadrature(P)

    p = atk["p_q"][rows, h_idx]
    b = atk["b_star"][rows, h_idx]
    E_loss = ap.eps_mean * b * TVL
    xi = xi_of_h(hg[h_idx], P)
    cov = aggregates["cap_scale"] * coverage(CC_val, xi, mp, tvl=TVL)
    L_nodes = eps_n[None, :] * b[:, None] * TVL[:, None]
    payout = (np.minimum(cov[:, None], L_nodes) * eps_w).sum(axis=-1)
    residual = E_loss - payout
    so = np.asarray(aggregates["sum_CC_others"])
    w_yield = CC_val / np.maximum(so + CC_val, 1e-9)
    ys = (1.0 - aggregates["gamma"]) * (1.0 - mp.phi) * aggregates["gross_rev"] * w_yield
    forfeit = p * CC_val if (mp.forfeiture and mp.insurance_on) else 0.0
    return (p * payout - forfeit + (1.0 - p) * ys
            - (mp.r_market / 4.0) * CC_val
            - (pp.c_h / 4.0) * hg[h_idx] * TVL
            - p * E_loss - rhoP * p * residual)
