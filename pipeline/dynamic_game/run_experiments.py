"""Run the full experiment suite for the dynamic-game addon.

Experiments
  1. Baseline dynamic equilibrium (MC over seeds).
  2. Moral hazard: full mechanism vs. no-forfeiture vs. no-insurance.
  3. eta sweep: how the value-elasticity of attack cost shapes who is
     attacked and the system loss rate (costs re-scaled to be equal at
     V_ref = $50M so only the SLOPE in V changes across eta).
  4. Frontier validation: re-estimate (kappa, eta) from simulated incidents
     with the FIRM two-margin quantile regression.
  5. Stage-Nash deviation check.

Writes outputs/results.json and figures to outputs/.
"""
import copy
import json
import time

import numpy as np

from params import AllParams
from game import run_game, run_mc
from analysis import (summarize, incidents_frame, epochs_frame,
                      frontier_regression, floor_check, deviation_check)
from attacker import attacker_best_response
from protocols import init_population
import figures

OUT = {}
T0 = time.time()
V_REF = 50.0
ETA_BASE = 0.20


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def rescaled(eta):
    """Params with eta changed but cost held fixed at V_ref (slope-only)."""
    P = AllParams()
    m = V_REF ** (ETA_BASE - eta)
    P.attacker.eta = eta
    P.attacker.gamma_c *= m
    P.attacker.F0 *= m
    return P


# ---------------- 1. Baseline ----------------
log("baseline MC (20 seeds)...")
P = AllParams()
runs = run_mc(P, n_seeds=20)
summ, ep, inc = summarize(runs, P)
OUT["baseline"] = summ
ep.to_csv("outputs/baseline_epochs.csv", index=False)
inc.to_csv("outputs/baseline_incidents.csv", index=False)
figures.fig_dynamics(ep, "outputs/fig_dynamics.png")
log(f"  hacks/yr={summ['hacks_per_year']:.2f}, "
    f"loss_rate={summ['loss_rate_bps']:.1f}bps, U={summ['mean_U']:.2f}, "
    f"gamma={summ['mean_gamma']:.2f}, h={summ['mean_h']:.2f}")

# ---------------- 2. Moral hazard regimes ----------------
log("moral-hazard regimes (20 seeds each)...")
regime_summaries = {}
for name, forfeiture, insurance in [("full_mechanism", True, True),
                                    ("no_forfeiture", False, True),
                                    ("no_insurance", True, False)]:
    Pr = AllParams()
    Pr.mech.forfeiture = forfeiture
    Pr.mech.insurance_on = insurance
    rr = run_mc(Pr, n_seeds=20)
    s, _, _ = summarize(rr, Pr)
    regime_summaries[name] = s
    log(f"  {name}: h={s['mean_h']:.3f}, hacks/yr={s['hacks_per_year']:.2f}, "
        f"raw_losses=${s['raw_losses_per_year']:.1f}M/yr")
OUT["moral_hazard"] = regime_summaries
figures.fig_moral_hazard(regime_summaries, "outputs/fig_moral_hazard.png")

# ---------------- 3. eta sweep ----------------
log("eta sweep (15 seeds each)...")
decile_profiles, loss_rates, eta_summ = {}, {}, {}
for eta in [0.0, 0.1, 0.2, 0.29, 0.5]:
    Pe = rescaled(eta)
    rr = run_mc(Pe, n_seeds=15)
    s, ep_e, _ = summarize(rr, Pe)
    eta_summ[eta] = s
    loss_rates[eta] = s["loss_rate_bps"]
    # attack-probability profile by TVL decile: per-protocol equilibrium
    # p_q at the final epoch, pooled across seeds
    probs, tvls = [], []
    for r in rr:
        probs.append(1 - (1 - r["final_p_q"]) ** 4)   # annualized
        tvls.append(r["pop"]["TVL"])
    probs = np.concatenate(probs); tvls = np.concatenate(tvls)
    dec = np.ceil(10 * (np.argsort(np.argsort(tvls)) + 1) / tvls.size)
    prof = np.array([probs[dec == d].mean() for d in range(1, 11)])
    decile_profiles[eta] = (np.arange(1, 11), prof)
    log(f"  eta={eta}: hacks/yr={s['hacks_per_year']:.2f}, "
        f"loss_rate={s['loss_rate_bps']:.1f}bps")
OUT["eta_sweep"] = {str(k): v for k, v in eta_summ.items()}
OUT["eta_decile_profiles"] = {str(k): [list(map(float, v[0])),
                                       list(map(float, v[1]))]
                              for k, v in decile_profiles.items()}
figures.fig_eta_sweep(decile_profiles, loss_rates, "outputs/fig_eta_sweep.png")

# ---------------- 4. Frontier validation ----------------
log("frontier validation on pooled baseline incidents...")
fr = frontier_regression(inc, tau=0.10)
fc = floor_check(inc, P)
OUT["frontier_validation"] = dict(fr, floor_check=fc,
                                  eta_calibrated=P.attacker.eta,
                                  kappa_calibrated=P.attacker.kappa)
figures.fig_frontier(inc, fr, P, "outputs/fig_frontier.png")
log(f"  eta_hat(rel)={fr['eta_hat_relative']:.3f}, "
    f"eta_hat(abs)={fr['eta_hat_absolute']:.3f}, "
    f"kappa_hat(rel)={fr['relative']['kappa_hat']:.2f} "
    f"[calibrated: eta={P.attacker.eta}, kappa={P.attacker.kappa}] "
    f"floor: {fc['frac_above_floor']:.1%} above")

# ---------------- 5. Equilibrium deviation check ----------------
log("stage-Nash deviation check (final epoch, 25 protocols)...")
rng = np.random.default_rng(99)
dev = deviation_check(P, runs[0], rng, n_check=25)
OUT["deviation_check"] = dev
log(f"  max unilateral gain={dev['max_gain']:.4g} $M "
    f"(rel {dev['rel_max_gain']:.2%})")

OUT["runtime_sec"] = round(time.time() - T0, 1)
with open("outputs/results.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)
log("done -> outputs/results.json")
