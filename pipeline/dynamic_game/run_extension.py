"""Endogenous-pool-alpha extension: experiments.

r_pool(C) = r_market + (r_pool0 - r_market)*K/(K+C). Sweep the alpha
capacity K against the fixed-alpha baseline (K = None):
  - carry-trade extinction (collateral/coverage, C*, r_pool path),
  - the deterrence side effect (equilibrium security h, hacks/yr),
  - moral-hazard robustness: forfeiture on/off under K = 1000.

Writes outputs/results_extension.json and outputs/fig_pool_alpha.png.
"""
import json
import time

import numpy as np

from params import AllParams
from game import run_game
from analysis import summarize, epochs_frame
import figures

T0 = time.time()
OUT = {}
SEEDS = 10
KS = [250.0, 1000.0, 4000.0, None]


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


panels = {}
for K in KS:
    P = AllParams()
    P.mech.pool_capacity = K
    rr = [run_game(P, seed=1000 + s) for s in range(SEEDS)]
    s, ep, _ = summarize(rr, P)
    key = "inf" if K is None else f"{K:.0f}"
    OUT[f"K_{key}"] = s
    g = ep.groupby("t")
    panels[key] = dict(
        t=(g.U.mean().index + 1).tolist(),
        r_pool=g.r_pool_eff.mean().tolist(),
        cc_to_cov=(g.sum_CC.mean() / g.coverage.mean()).tolist(),
        mean_h=g.mean_h.mean().tolist(),
        hacks_yr=float(s["hacks_per_year"]),
        eps_max=float(ep.eps_stage.max()),
    )
    log(f"K={key}: C_total={s['mean_C_total']:.0f} CC/cov="
        f"{s['collateral_to_coverage']:.3f} r_pool_fin={s['final_r_pool']:.4f} "
        f"h={s['mean_h']:.2f} hacks/yr={s['hacks_per_year']:.2f} "
        f"eps_max={panels[key]['eps_max']:.2e}")

log("moral-hazard robustness under K=1000 (forfeiture off)...")
P = AllParams()
P.mech.pool_capacity = 1000.0
P.mech.forfeiture = False
rr = [run_game(P, seed=1000 + s) for s in range(SEEDS)]
s_nf, _, _ = summarize(rr, P)
OUT["K_1000_no_forfeiture"] = s_nf
log(f"  no_forfeiture@K1000: h={s_nf['mean_h']:.3f}, "
    f"hacks/yr={s_nf['hacks_per_year']:.2f} "
    f"(with forfeiture: h={OUT['K_1000']['mean_h']:.3f}, "
    f"{OUT['K_1000']['hacks_per_year']:.2f})")

figures.fig_pool_alpha(panels, "outputs/fig_pool_alpha.png")
OUT["runtime_sec"] = round(time.time() - T0, 1)
with open("outputs/results_extension.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)
log("done -> outputs/results_extension.json")
