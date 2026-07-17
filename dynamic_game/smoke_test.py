"""Smoke test: one full game run + timing + key diagnostics to JSON."""
import json
import time

import numpy as np

from params import AllParams
from game import run_game
from analysis import summarize, floor_check, incidents_frame

P = AllParams()
t0 = time.time()
run = run_game(P, seed=42)
elapsed = time.time() - t0

out, ep, inc = summarize([run], P)
out["elapsed_sec"] = round(elapsed, 2)
out["fp_iters_by_epoch"] = [e["fp_iters"] for e in run["epochs"]]
out["mean_h_by_epoch"] = [round(e["mean_h"], 3) for e in run["epochs"]]
out["U_by_epoch"] = [round(e["U"], 2) for e in run["epochs"]]
out["gamma_by_epoch"] = [round(e["gamma"], 3) for e in run["epochs"]]
out["CLP_by_epoch"] = [round(e["CLP"], 1) for e in run["epochs"]]
out["insured_frac_by_epoch"] = [round(e["insured_frac"], 3) for e in run["epochs"]]
if len(inc):
    out["floor_check"] = floor_check(inc, P)

with open("outputs/smoke_test.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
