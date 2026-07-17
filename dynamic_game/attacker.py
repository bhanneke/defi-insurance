"""Strategic attacker: Becker target selection with the FIRM cost function.

Implements Sec. 3 of the FIRM paper:
    beta(e, h) = (1 - exp(-a*e)) * (1+h)^(-zeta)     (extraction technology)
    c(e, V)    = gamma_c * e^kappa * V^eta           (structural cost curve)
    pi(e)      = E[eps] * beta(e, h) * V - c(e, V)
    e* = argmax_{e >= e_min} pi(e); attack iff pi* > 0 (Prop. 1),
with the extensive margin smoothed by a random outside option
F ~ Exp(F_bar): P(exec | opportunity) = 1 - exp(-pi*/F_bar).

All functions are vectorized over numpy arrays of targets.
"""
import numpy as np

from params import AttackerParams


def beta_extraction(e, h, ap: AttackerParams):
    """Blast radius technology: concave in effort, decreasing in defense."""
    return (1.0 - np.exp(-ap.a_eff * e)) * (1.0 + h) ** (-ap.zeta_def)


def attack_cost(e, V, ap: AttackerParams):
    """c(e,V) = (F0 + gamma_c e^kappa) V^eta.

    The fixed component F0 (target identification / preparation) also scales
    with V^eta, so d log c / d log V = eta exactly, matching the FIRM
    structural estimate; the effort elasticity approaches kappa for e >> 1.
    """
    return (ap.F0 + ap.gamma_c * e ** ap.kappa) * V ** ap.eta


def _foc(e, V, h, ap: AttackerParams):
    """d pi / d e. Positive left of the interior optimum, negative right."""
    marg_gain = (ap.eps_mean * ap.a_eff * np.exp(-ap.a_eff * e)
                 * (1.0 + h) ** (-ap.zeta_def) * V)
    marg_cost = ap.gamma_c * ap.kappa * e ** (ap.kappa - 1.0) * V ** ap.eta
    return marg_gain - marg_cost


def solve_effort(V, h, ap: AttackerParams, n_bisect: int = 70):
    """Vectorized optimal effort e*(V, h) on [e_min, e_max].

    The FOC has a unique interior root because marginal gain is decreasing
    while marginal cost is increasing (kappa > 1). Bisection in log-effort.
    """
    V = np.asarray(V, dtype=float)
    h = np.asarray(h, dtype=float)
    lo = np.full(np.broadcast(V, h).shape, np.log(ap.e_min))
    hi = np.full_like(lo, np.log(ap.e_max))

    # Corner: already past the optimum at e_min
    at_min = _foc(ap.e_min, V, h, ap) <= 0.0

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        pos = _foc(np.exp(mid), V, h, ap) > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)

    e_star = np.exp(0.5 * (lo + hi))
    e_star = np.where(at_min, ap.e_min, e_star)
    return e_star


def attacker_best_response(V, h, ap: AttackerParams):
    """Full attacker solution for target arrays (V, h).

    Returns dict with:
      e_star  : optimal effort
      b_star  : expected blast radius beta(e*, h)
      pi_star : expected attacker profit ($M); attack iff > 0 (participation)
      p_exec  : P(attack executed | opportunity arrives)
      p_q     : quarterly attack probability = q_opp * p_exec
      lam_ann : implied annualized per-protocol hazard (for the pricing layer)
    """
    e_star = solve_effort(V, h, ap)
    b_star = beta_extraction(e_star, h, ap)
    pi_star = ap.eps_mean * b_star * V - attack_cost(e_star, V, ap)

    p_exec = np.where(pi_star > 0.0, 1.0 - np.exp(-np.maximum(pi_star, 0.0) / ap.F_bar), 0.0)
    p_q = ap.q_opp * p_exec
    p_q = np.clip(p_q, 0.0, 0.999)
    lam_ann = -np.log1p(-p_q) / 0.25
    return dict(e_star=e_star, b_star=b_star, pi_star=pi_star,
                p_exec=p_exec, p_q=p_q, lam_ann=lam_ann)


def participation_threshold(h, ap: AttackerParams, v_grid=None):
    """V*(h): smallest V with pi* > 0 (Prop. 1's rising threshold).

    Numerical, for diagnostics/calibration.
    """
    if v_grid is None:
        v_grid = np.logspace(-1, 4, 2000)  # $0.1M .. $10B
    h = np.atleast_1d(np.asarray(h, dtype=float))
    out = np.full(h.shape, np.nan)
    for i, hi in enumerate(h):
        br = attacker_best_response(v_grid, np.full_like(v_grid, hi), ap)
        idx = np.argmax(br["pi_star"] > 0.0)
        if br["pi_star"][idx] > 0.0:
            out[i] = v_grid[idx]
    return out
