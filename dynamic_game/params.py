"""Parameters for the dynamic-game addon.

Combines:
- Insurance mechanism parameters from the MARBLE2026 paper
  ("Decentralized Finance: A Market Mechanism for Cybersecurity Risk
  Insurance"), Appendix B, Table 2.
- Attacker-economics primitives from the FIRM paper ("How Attacker
  Economics Shape Security Outcomes in Decentralized Infrastructure"),
  Section 3 (Mechanism) and Section 5.4 (structural estimates):
      c(e, V) = gamma_c * e^kappa * V^eta,
      eta_hat in [0.20, 0.29]  (frontier reading ~0.20),
      kappa_hat ~ 2.7 (attack length) / 3.1 (distinct techniques).

All money amounts in $M. Time unit: one epoch = one quarter.
This module is exploratory addon code; it is NOT part of either paper's
pipeline.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AttackerParams:
    """Becker-style attacker (FIRM paper, Sec. 3).

    Profit: pi(e) = E[eps] * beta(e, h) * V - c(e, V)
      beta(e, h) = (1 - exp(-a_eff * e)) * (1 + h)^(-zeta_def)
      c(e, V)    = gamma_c * e^kappa * V^eta
    Effort e is measured in 'attacker actions' (the empirical attack-length
    proxy), e >= e_min = 1 (a viable exploit chain has at least one action).

    Extensive margin: conditional on an attack opportunity (prob q_opp per
    quarter), the attack is executed iff pi* exceeds the attacker's outside
    option F ~ Exponential(F_bar):  P(exec) = 1 - exp(-pi*/F_bar), pi* > 0.
    This smooths Prop. 1's threshold V*(h) into an attack probability that
    rises with V and falls with h, as in the data.
    """
    eta: float = 0.20         # value-elasticity of attack cost (structural est.)
    kappa: float = 2.7        # effort-elasticity of attack cost (attack length)
    gamma_c: float = 0.002    # variable cost scale; calibrated
    F0: float = 0.8           # fixed preparation cost (also scales with V^eta):
                              # c(e,V) = (F0 + gamma_c e^kappa) V^eta. Keeps the
                              # FIRM value-elasticity eta exact while letting the
                              # participation threshold V*(h) be calibrated
                              # independently of the effort margin.
    e_min: float = 1.0        # minimum viable effort (actions)
    e_max: float = 400.0      # numerical upper bound for effort search
    a_eff: float = 0.12       # blast-radius saturation rate in effort
    zeta_def: float = 1.25    # defense elasticity of extraction
    q_opp: float = 0.04       # quarterly prob. an opportunity/probe arrives
    F_bar: float = 25.0       # mean attacker outside option ($M); calibrated
    eps_a: float = 8.0        # Beta(a,b) execution/extraction noise on loss
    eps_b: float = 2.0        # -> mean eps = 0.8

    @property
    def eps_mean(self) -> float:
        return self.eps_a / (self.eps_a + self.eps_b)


@dataclass
class MechanismParams:
    """Insurance mechanism (MARBLE2026 paper, Table 2 baseline)."""
    mu: float = 3.0           # coverage scale, Eq. (2)
    theta: float = 0.50       # coverage concavity, Eq. (2)
    phi: float = 0.01         # operator fee
    alpha: float = 0.6        # weight on utilization in gamma, Eq. (8)
    beta_u: float = 1.0       # utilization convexity in gamma
    delta: float = 0.7        # risk-price convexity in gamma
    omega: Tuple[float, float, float, float] = (0.40, 0.30, 0.20, 0.10)
    U_target: float = 15.0
    U_min: float = 1.0        # prudential cap bounds, Eq. (5)
    U_max_hi: float = 30.0
    kappa_U: float = 100.0    # dynamic-cap sensitivity (dynamic cap ON)
    r_market: float = 0.05    # annual
    r_pool: float = 0.10      # annual
    rho_LP: float = 0.005     # annual LP risk premium
    kappa_LP: float = 2.0     # LP capital adjustment speed, Prop. 2(ii)
    fee_base_annual: float = 0.03
    fee_hack_jump: float = 0.10   # extra fee yield, applied 1 day per hack
    eta_blend: float = 0.5    # gamma_sim blending (paper App. B); renamed
                              # from the paper's 'eta' to avoid clashing with
                              # the attacker cost elasticity
    forfeiture: bool = True   # collateral forfeited to pool on hack
    insurance_on: bool = True # False = no-insurance counterfactual


@dataclass
class ProtocolParams:
    """Protocol population and the (new) security-choice margin."""
    n_protocols: int = 500
    # TVL ~ 5*(1+Pareto(1.2)), clipped [2, 10000] $M (v6 convention,
    # matches paper App. B: heavy-tailed, median ~$9M)
    tvl_scale: float = 5.0
    tvl_pareto_a: float = 1.2
    tvl_clip: Tuple[float, float] = (2.0, 10_000.0)
    rho_med: float = 1.5      # protocol risk aversion: lognormal median
    rho_sigma: float = 0.5
    rho_clip: Tuple[float, float] = (0.0, 3.0)
    # security choice h >= 0 (index). Annual cost: c_h * h * TVL.
    c_h: float = 0.004        # 0.4% of TVL per year per unit of h
    h_max: float = 4.0
    n_h_grid: int = 13
    # coverage security multiplier xi(h) = xi0 + xi1*h, clipped to (0, xi_cap]
    xi0: float = 0.2
    xi1: float = 0.15
    xi_cap: float = 1.0       # paper Table 2: xi in (0, 1]
    n_cc_grid: int = 24
    cc_tvl_cap: float = 0.5   # collateral cannot exceed 50% of TVL


@dataclass
class GameParams:
    n_quarters: int = 8       # 2 years, matching the paper's horizon
    CLP0: float = 250.0       # initial LP capital ($M)
    tvl_growth: float = 0.10  # annual TVL drift
    fp_max_iter: int = 40     # within-epoch fixed-point iterations
    fp_tol: float = 1e-4      # sup-norm tolerance on (p, gamma)
    fp_damping: float = 0.5   # damping on the price iterate
    CLP_floor: float = 10.0
    seed: int = 1234


@dataclass
class AllParams:
    attacker: AttackerParams = field(default_factory=AttackerParams)
    mech: MechanismParams = field(default_factory=MechanismParams)
    proto: ProtocolParams = field(default_factory=ProtocolParams)
    game: GameParams = field(default_factory=GameParams)
