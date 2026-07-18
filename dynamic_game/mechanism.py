"""Insurance mechanism equations (MARBLE2026 paper, Sec. 4).

Ported to match the paper's equations; helper structure follows
defi_insurance_simulation_PAPER_ALIGNED_v6.py where the paper delegates
implementation detail to the simulation appendix (gamma blending, fees).
"""
import numpy as np

from params import MechanismParams


def coverage(CC, xi, mp: MechanismParams, tvl=None):
    """Eq. (2): coverage = mu * CC^theta * (1 + xi); optionally capped at TVL."""
    cov = mp.mu * np.maximum(CC, 0.0) ** mp.theta * (1.0 + xi)
    if tvl is not None:
        cov = np.minimum(cov, tvl)
    return cov


def utilization(total_coverage, CLP):
    """Eq. (3)."""
    return total_coverage / max(CLP, 1e-9)


def hazard_from_term_structure(P_hack_vec, T_vec=(0.25, 0.5, 0.75, 1.0),
                               lam_hi=2.0, n_grid=4001):
    """Eq. (4): least-squares constant hazard from the four expiries.

    Wider grid than v6 (endogenous hazards can exceed its 0.05 cap).
    """
    P = np.asarray(P_hack_vec, dtype=float)
    lam_grid = np.linspace(0.0, lam_hi, n_grid)
    diffs = (1.0 - np.exp(-np.outer(lam_grid, T_vec))) - P[None, :]
    return float(lam_grid[np.argmin(np.square(diffs).sum(axis=1))])


def term_structure_from_hazard(lam_ann, T_vec=(0.25, 0.5, 0.75, 1.0)):
    return 1.0 - np.exp(-lam_ann * np.asarray(T_vec))


def p_risk_anchor(P_hack_vec, mp: MechanismParams):
    """Eqs. (6) and (7): near-term index and annualized anchor."""
    lam = hazard_from_term_structure(P_hack_vec)
    p_anchor = 1.0 - np.exp(-lam)
    w = np.asarray(mp.omega, dtype=float)
    w = w / w.sum()
    p_risk = float((w * np.asarray(P_hack_vec)).sum())
    return lam, p_anchor, p_risk


def U_max_dynamic(p_anchor, mp: MechanismParams):
    """Eq. (5): prudential cap tightens with market-implied annual risk."""
    return mp.U_min + (mp.U_max_hi - mp.U_min) / (1.0 + mp.kappa_U * p_anchor)


def gamma_raw(U, p_risk, p_anchor, mp: MechanismParams):
    """Eq. (8), clipped to [0, 1]."""
    rU = (U / max(mp.U_target, 1e-12)) ** mp.beta_u
    rP = (p_risk / max(p_anchor, 1e-12)) ** mp.delta
    return float(np.clip(mp.alpha * rU + (1.0 - mp.alpha) * rP, 0.0, 1.0))


def gamma_blended(g_raw, CLP, sum_CC, mp: MechanismParams):
    """Paper App. B: blend Eq. (8) with the capital-proportional fair split."""
    g_fair = CLP / max(CLP + sum_CC, 1e-12)
    g = g_fair + mp.eta_blend * (g_raw - g_fair)
    return float(np.clip(g, 0.0, 1.0)), float(g_fair)


def pool_return(C_total, mp: MechanismParams):
    """Annual pool return; endogenous scarce alpha when pool_capacity is set.

    r_pool(C) = r_market + (r_pool0 - r_market) * K / (K + C):
    continuous, nonincreasing, -> r_market as C grows. Total revenue
    R(C) = r_pool(C)*C is nondecreasing and concave with R(0)=0 (the
    property the equilibrium existence lemma needs; see PROOF_NOTES.md).
    """
    if mp.pool_capacity is None:
        return mp.r_pool
    K = mp.pool_capacity
    # Fact 1 (R increasing & concave) requires positive excess alpha and
    # positive capacity; fail loudly on misconfiguration.
    assert K > 0 and mp.r_pool > mp.r_market, \
        "pool_capacity requires K > 0 and r_pool > r_market"
    return mp.r_market + (mp.r_pool - mp.r_market) * K / (K + max(C_total, 0.0))


def quarterly_pool_revenue(C_total, n_hack_events, mp: MechanismParams):
    """Base yield plus speculator fee inflows over one quarter.

    v6 convention: fee yields applied to total capital; the hack-day fee
    jump applies for one day per hack event.
    """
    base = (pool_return(C_total, mp) / 4.0) * C_total
    fees = (mp.fee_base_annual / 4.0) * C_total \
        + (mp.fee_hack_jump / 365.0) * C_total * n_hack_events
    return base, fees
