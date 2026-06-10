# DeFi Insurance: A Market Mechanism for Cybersecurity Risk Insurance

Simulation code for the paper:

> Hanneke, B.: *Decentralized Finance: A Market Mechanism for Cybersecurity Risk Insurance.* Goethe University Frankfurt.

The paper proposes a two-layer market mechanism for on-chain cybersecurity risk transfer: binary HACK/NOHACK prediction markets provide continuous, market-implied hack probabilities (insurance pricing layer), while protocols post forfeitable collateral to access coverage from a liquidity-provider capital pool whose yield share adjusts dynamically to utilization and market-priced risk (insurance provision layer).

## Contents

| Path | Description |
|---|---|
| `defi_insurance_simulation.py` | Monte-Carlo simulation behind Section 6 ("Stylized Simulation") and Appendix B of the paper |
| `outputs/` | Reference outputs (figures and run-level metrics) |
| `archive/legacy-2025-07/` | Code of an earlier working-paper iteration (loss-given-hack strike tokens, premium payments, Stackelberg operator analysis). Kept for provenance only — it does **not** correspond to the published paper. |

## Running the simulation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python defi_insurance_simulation.py
```

The baseline configuration reproduces the paper: 1000 Monte-Carlo runs, 500 heterogeneous protocols, a 2-year horizon with daily steps. Runs are seeded (base seed 1234, per-run offsets), so results are deterministic. A full 1000-run simulation takes a while; set `n_mc_runs = 200` in `SimulationParams` for a quick pass.

Outputs are written to `outputs/`: the summary dashboard (`simulation_results.png`, Fig. 3), run-level distribution boxplots (`boxplots_across_runs.png/.pdf`, Fig. 2), the protocol population (`protocol_distribution_hist.png`, Fig. 4; `protocol_distribution.csv`), per-run metrics (`runlevel_metrics.csv`), and pickled raw runs (`runs.pkl`, `summary.pkl` — large, git-ignored). Headline metrics (average utilization, yield share, covered-dollar-years, loss rate, realized LP/protocol APYs, cumulative yield split) are printed to stdout.

## Code-to-paper mapping

| Paper | Code |
|---|---|
| Eq. (2) coverage function | `protocol_target_CC`, coverage computation in `run_single_simulation` step 3 |
| Eq. (3) utilization | `run_single_simulation` step 4 |
| Eq. (4) hazard-rate estimation from the HACK term structure | `infer_lambda_from_term_structure` |
| Eq. (5) prudential cap U_max(t) | `Umax_from_paper` (baseline uses the fixed cap, `kappa_Ucap = 0`) |
| Eqs. (6)–(7) risk index and anchor | `compute_p_anchor_p_risk` |
| Eq. (8) yield-share γ | `gamma_from_paper` |
| Appendix B γ_fair blending (η = 0.5) | `run_single_simulation` step 6 |
| Prop. 2(ii) LP capital adjustment | `run_single_simulation` steps 7–8 |
| Table 2 baseline parameters | `SimulationParams` |
| Protocol population (Pareto TVL, risk aversion, security multipliers) | `init_protocol_population` |
| Hack arrivals and payout waterfall (collateral burns first, then LP pool) | `run_single_simulation` step 5 |

## Requirements

Python ≥ 3.9 with `numpy`, `pandas`, `matplotlib` (see `requirements.txt`).
