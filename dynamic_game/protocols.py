"""Protocol population and best responses.

Protocols choose collateral C_C (continuous, exact best response) AND a
security level h (grid), anticipating the strategic attacker's response
p_q(V, h) and b*(V, h) from the FIRM model.

Objective (quarterly, $M), extending Eq. (7) of the insurance paper:

  U(C_C, h) =  p * E[min(cov, L)]              expected insurance payout
             - 1{forfeiture} * p * C_C          collateral forfeited on hack
             + (1-p) * yield_share(C_C)         protocol share of pool yield
             - (r_market/4) * C_C               collateral opportunity cost
             - (c_h/4) * h * TVL                security expenditure
             - p * E[L]                         expected raw loss
             - rho_P * p * E[L - min(cov, L)]   risk-aversion penalty on the
                                                UNINSURED residual

L = eps * b*(V,h) * V with execution noise eps ~ Beta(eps_a, eps_b).

The C_C margin is concave for fixed h: the payout term is concave
(theta < 1), forfeiture/opportunity costs are linear, and the yield term
is concave by Lemma 1 of PROOF_NOTES.md (R(C) concave through the origin).
Hence golden-section search returns the EXACT continuous best response,
which is what lets the no-deviation check certify the equilibrium at a
strict tolerance instead of at grid resolution.
"""
import numpy as np
from scipy import stats

from params import AllParams
from attacker import attacker_best_response
from mechanism import coverage

_INVPHI = (np.sqrt(5.0) - 1.0) / 2.0


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


def utility_cc(CC, pop, atk, aggregates, P: AllParams, eps_n, eps_w):
    """Protocol utility for candidate collateral CC, shape (n, n_h):
    one candidate per (protocol, security level). Returns (n, n_h)."""
    mp, pp, ap = P.mech, P.proto, P.attacker
    TVL, rhoP = pop["TVL"], pop["rhoP"]
    hg = atk["h_grid"]
    p = atk["p_q"]                                        # (n, n_h)
    b = atk["b_star"]
    E_loss = ap.eps_mean * b * TVL[:, None]
    xi = xi_of_h(hg, P)[None, :]
    cov = aggregates["cap_scale"] * coverage(CC, xi, mp, tvl=TVL[:, None])
    L_nodes = eps_n[None, None, :] * b[..., None] * TVL[:, None, None]
    payout = (np.minimum(cov[..., None], L_nodes) * eps_w).sum(axis=-1)
    residual = E_loss - payout
    so = np.asarray(aggregates["sum_CC_others"])[:, None]
    w_yield = CC / np.maximum(so + CC, 1e-9)
    ys = (1.0 - aggregates["gamma"]) * (1.0 - mp.phi) \
        * aggregates["gross_rev"] * w_yield
    forfeit = (p * CC) if (mp.forfeiture and mp.insurance_on) else 0.0
    return (p * payout - forfeit + (1.0 - p) * ys
            - (mp.r_market / 4.0) * CC
            - (pp.c_h / 4.0) * hg[None, :] * TVL[:, None]
            - p * E_loss - rhoP[:, None] * p * residual)


def _golden_cc(pop, atk, aggregates, P: AllParams, eps_n, eps_w,
               n_iter: int = 60):
    """Vectorized golden-section max over CC in [0, cap] for every
    (protocol, h) cell. Exact for concave objectives (Lemma 1)."""
    n = pop["TVL"].shape[0]
    n_h = atk["h_grid"].size
    lo = np.zeros((n, n_h))
    hi = np.broadcast_to((P.proto.cc_tvl_cap * pop["TVL"])[:, None],
                         (n, n_h)).copy()

    def U(x):
        return utility_cc(x, pop, atk, aggregates, P, eps_n, eps_w)

    x1 = hi - _INVPHI * (hi - lo)
    x2 = lo + _INVPHI * (hi - lo)
    f1, f2 = U(x1), U(x2)
    for _ in range(n_iter):
        left = f1 >= f2
        lo = np.where(left, lo, x1)
        hi = np.where(left, x2, hi)
        x1_new = np.where(left, hi - _INVPHI * (hi - lo), x2)
        x2_new = np.where(left, x1, lo + _INVPHI * (hi - lo))
        # one fresh evaluation per iteration; the surviving interior point
        # carries its known value: on 'left' old x1 becomes the new x2,
        # on 'right' old x2 becomes the new x1.
        x_eval = np.where(left, x1_new, x2_new)
        f_eval = U(x_eval)
        f1_old = f1
        f1 = np.where(left, f_eval, f2)
        f2 = np.where(left, f1_old, f_eval)
        x1, x2 = x1_new, x2_new
    CC = 0.5 * (lo + hi)
    u = U(CC)
    u0 = U(np.zeros_like(CC))
    pick0 = u0 >= u
    return np.where(pick0, 0.0, CC), np.where(pick0, u0, u)


def best_response(pop, atk, aggregates, P: AllParams):
    """Joint best response: exact continuous CC (golden section) x h grid.

    aggregates: dict with
      gamma         : blended LP yield share
      gross_rev     : quarterly pool revenue (base + fees), $M
      sum_CC_others : (n,) posted collateral of the OTHER protocols
      cap_scale     : prudential-cap coverage scaling, min(1, U_max/U)
    """
    hg = atk["h_grid"]
    n = pop["TVL"].shape[0]
    eps_n, eps_w = eps_quadrature(P)

    if P.mech.insurance_on:
        CC_star, u = _golden_cc(pop, atk, aggregates, P, eps_n, eps_w)
    else:
        CC_star = np.zeros((n, hg.size))
        u = utility_cc(CC_star, pop, atk, aggregates, P, eps_n, eps_w)

    ih = np.argmax(u, axis=1)
    rows = np.arange(n)
    return dict(
        h=hg[ih], CC=CC_star[rows, ih], ih=ih,
        p_q=atk["p_q"][rows, ih], b_star=atk["b_star"][rows, ih],
        e_star=atk["e_star"][rows, ih], pi_star=atk["pi_star"][rows, ih],
        xi=xi_of_h(hg[ih], P), utility=u[rows, ih],
    )
