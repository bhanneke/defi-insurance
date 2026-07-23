"""Calibration probe for attacker primitives.

Targets (see README):
  - participation threshold V*(h=0) in the low single-digit $M range
    (tiny protocols are not worth attacking),
  - median optimal effort ~5-20 attacker actions (FIRM attack-length range),
  - median expected blast radius roughly 0.2-0.7,
  - aggregate ~2-6 executed hacks/year in a 500-protocol universe,
  - median loss per covered hack in the $5-15M range (paper Sec. 6: ~$8M).

Writes a moment table to outputs/calibration.csv for chosen (gamma_c, F_bar).
"""
import itertools
import numpy as np
import pandas as pd

from params import AllParams
from attacker import attacker_best_response, participation_threshold
from protocols import init_population


def moments(P: AllParams, seed=7):
    rng = np.random.default_rng(seed)
    pop = init_population(rng, P)
    V = pop["TVL"]
    h0 = np.zeros_like(V)          # no defense (upper bound on attack activity)
    h1 = np.full_like(V, 1.5)      # a mid-grid defense level
    out = {}
    for tag, h in [("h0", h0), ("h15", h1)]:
        br = attacker_best_response(V, h, P.attacker)
        sel = br["pi_star"] > 0
        hacks_yr = 4.0 * br["p_q"].sum()
        eps_m = P.attacker.eps_mean
        exp_hack_w = br["p_q"] / max(br["p_q"].sum(), 1e-12)
        med_L = np.interp(0.5, np.cumsum(exp_hack_w[np.argsort(eps_m * br["b_star"] * V)]),
                          np.sort(eps_m * br["b_star"] * V))
        out.update({
            f"{tag}_attackable_frac": float(sel.mean()),
            f"{tag}_hacks_per_year": float(hacks_yr),
            f"{tag}_med_effort": float(np.median(br["e_star"][sel])) if sel.any() else np.nan,
            f"{tag}_med_blast": float(np.median(br["b_star"][sel])) if sel.any() else np.nan,
            f"{tag}_med_loss_hackw": float(med_L),
        })
    out["Vstar_h0"] = float(participation_threshold([0.0], P.attacker)[0])
    out["Vstar_h2"] = float(participation_threshold([2.0], P.attacker)[0])
    return out


def main():
    rows = []
    for gc, F0, Fb in itertools.product([0.001, 0.002, 0.004],
                                        [0.2, 0.4, 0.8, 1.5],
                                        [12.0, 25.0, 50.0]):
        P = AllParams()
        P.attacker.gamma_c = gc
        P.attacker.F0 = F0
        P.attacker.F_bar = Fb
        m = moments(P)
        rows.append(dict(gamma_c=gc, F0=F0, F_bar=Fb, **m))
    df = pd.DataFrame(rows)
    df.to_csv("outputs/calibration.csv", index=False)
    cols = ["gamma_c", "F0", "F_bar", "Vstar_h0", "Vstar_h2",
            "h0_attackable_frac", "h0_hacks_per_year", "h15_hacks_per_year",
            "h0_med_effort", "h0_med_blast", "h0_med_loss_hackw"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
