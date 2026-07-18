# Dynamic Game: Endogenous Attacker for the Web3 Insurance Mechanism

**Exploratory addon — not part of either paper's pipeline.**

Combines two papers:

1. **Insurance mechanism** — *"Decentralized Finance: A Market Mechanism for
   Cybersecurity Risk Insurance"* (MARBLE2026 / PISA version). Equation and
   parameter references (Eqs. 2–8, Table 2) follow the MARBLE2026 tex.
2. **Attacker economics** — *"How Attacker Economics Shape Security Outcomes
   in Decentralized Infrastructure"* (Attacker_Economics_FIRM repo). The loss
   process is built from its Becker target-selection model and structural
   estimates.

## What changes relative to the paper simulation

The paper's stylized simulation (`defi_insurance_simulation_PAPER_ALIGNED_v6.py`)
draws hack arrivals from an **exogenous** Gamma–Poisson hazard and loss
severities from a distribution. Here both are **equilibrium objects**:

- A strategic attacker observes every protocol's value at stake `V` (TVL) and
  security level `h`, chooses attack effort `e`, and attacks only when
  profitable (participation constraint).
- Protocols choose collateral `C_C` **and** security `h`, anticipating the
  attacker's response.
- Speculators price HACK term structures under rational expectations: implied
  hazards equal the attack probabilities the attacker's strategy actually
  induces (the truth in "truthful risk assessment" is endogenous).
- LP capital follows the Prop. 2(ii) adjustment dynamics.

## Loss function (from the FIRM paper)

```
beta(e, h) = (1 - exp(-a*e)) * (1+h)^(-zeta)      extraction technology
c(e, V)    = (F0 + gamma_c * e^kappa) * V^eta      attack cost
pi(e)      = E[eps] * beta(e,h) * V - c(e,V)       attacker profit
e* = argmax pi,  attack iff pi* > 0                participation (Prop. 1)
L  = eps * beta(e*,h) * V,  eps ~ Beta(8,2)        realized loss
```

- `eta = 0.20`, `kappa = 2.7`: the FIRM structural estimates (value-elasticity
  of attack cost; effort-elasticity via the attack-length proxy).
- The fixed preparation cost `F0` also scales with `V^eta`, so
  `dlog c / dlog V = eta` **exactly** as estimated, while `F0` pins the
  participation threshold `V*(h)` independently of the effort margin.
- Extensive margin smoothed by a stochastic attacker outside option
  `F ~ Exp(F_bar)`: `P(attack | opportunity) = 1 - exp(-pi*/F_bar)`;
  opportunities arrive per protocol-quarter with probability `q_opp`.

Calibration targets (see `calibrate.py`, `outputs/calibration.csv`):
`V*(h=0) ≈ $3M`, median effort 5–20 actions, ~2–6 executed hacks/year in a
500-protocol universe, median loss per hack in the $5–15M range.

## Solution concept (honest statement)

Within each quarter: stage Nash equilibrium via damped fixed-point iteration
(attacker best response is computed once per epoch since it depends only on
`(V, h)`; protocol best responses are exact grid argmaxes over `(h, C_C)`;
prices and mechanism aggregates update until convergence). Across quarters:
state (TVL, LP capital) transitions and the stage game repeats — i.e., a
Markov game with **myopic** per-quarter payoffs, not a full MPE with
forward-looking value functions. Convergence is checked every epoch and a
sampled no-unilateral-deviation check runs at the end (`deviation_check`).

## Files

| File | Contents |
|---|---|
| `params.py` | All parameters, both papers' calibrations documented |
| `attacker.py` | Effort choice, participation, attack probabilities |
| `mechanism.py` | Eqs. 2–8: coverage, utilization, hazard fit, dynamic cap, yield share |
| `protocols.py` | Population, joint `(h, C_C)` best response, deviation payoffs |
| `game.py` | Stage fixed point, realization, settlement, LP dynamics |
| `pricing` | inside `game.py`: RE term structures from the endogenous hazard |
| `analysis.py` | Summaries, frontier re-estimation, floor/deviation checks |
| `figures.py` | The four figures (palette validated per dataviz skill) |
| `calibrate.py` | Attacker-primitive calibration probe |
| `smoke_test.py` | One-run diagnostic |
| `run_experiments.py` | Full suite -> `outputs/results.json` + figures |
| `PROOF_NOTES.md` | Equilibrium proofs for the endogenous-alpha extension |
| `prove_equilibrium.py` | Computational certification (VERDICT output) |
| `run_extension.py` | Endogenous-alpha experiments -> `results_extension.json` |

Run: `python3 run_experiments.py` (stock python3 with numpy/pandas/scipy/
statsmodels/matplotlib; ~45 s total).

## Headline results (outputs/results.json, 2026-07-17)

1. **The endogenous game lands where the paper's calibrated simulation
   points.** Baseline: 2.3 hacks/yr, median loss $9.7M (paper: ~$8M), loss
   rate ~56 bps/yr (paper: 76 bps), mean utilization 9.0 (paper: 9.94), no
   insolvencies. None of these moments were targeted directly.
2. **The collateral lever is what stands between the mechanism and
   attacker-amplified moral hazard.** Removing forfeiture (coverage without
   penalty): equilibrium security collapses 1.16 -> 0.18, hacks/yr 2.3 -> 10.8,
   raw losses $71M -> $163M/yr. With forfeiture, security matches the
   uninsured benchmark (1.16 vs 1.14) — the penalty fully restores incentives;
   residual moral hazard is ~10% on the attack rate (2.30 vs 2.08). With a
   *strategic* attacker the moral-hazard externality is 5x on occurrence, far
   larger than a fixed-hazard analysis would suggest.
3. **eta governs who gets attacked.** With costs equalized at V=$50M, pure
   Becker (`eta=0`) concentrates attacks almost entirely on the top TVL decile;
   `eta=0.5` spreads them down the size distribution. At the estimated
   `eta≈0.2`, mid-size protocols see materially positive attack probability —
   relevant for the mechanism's adverse-selection story.
4. **The participation frontier re-emerges and is estimable.** 100% of
   simulated incidents respect the theoretical floor; the FIRM two-margin
   quantile regression on simulated equilibrium data recovers
   `eta_hat = 0.14` (calibrated 0.20) — downward-biased exactly as the FIRM
   limitations section predicts for frontier estimates under selection and
   heterogeneous defense (the sim reproduces the lower-bound property).
5. **Mechanism-design insight: protocols yield-farm the pool.** Under the full
   Eq. (7) objective with `r_pool > r_market` and a protocol yield share
   pro-rata in collateral, rational protocols post collateral far beyond the
   coverage FOC (up to the wealth cap; collateral/coverage ~75% vs the paper's
   ~11%). Solvency and caps hold, but if the operator wants collateral to be a
   *penalty* margin rather than a carry trade, the protocol yield share should
   key on coverage, not collateral. This only appears when protocols optimize
   the full objective — the paper's simulation pins collateral via the
   coverage FOC alone.

## Extension: endogenous pool alpha (carry-trade fix)

Motivated by finding 5: make the pool's excess return scarce,
`r_pool(C) = r_market + (r_pool0 - r_market) * K/(K + C)` with capacity K
(`params.MechanismParams.pool_capacity`; None = fixed-alpha baseline).
Equilibrium theory in `PROOF_NOTES.md`, certified by
`prove_equilibrium.py`; experiments in `run_extension.py`
(fig_pool_alpha.png, results_extension.json, 10 seeds per K):

| | K=250 | K=1000 | K=4000 | fixed |
|---|---|---|---|---|
| pool capital C* ($M) | 2489 | 3258 | 4192 | 5058 |
| collateral/coverage | 0.54 | 0.61 | 0.68 | 0.75 |
| final r_pool | 5.4% | 6.0% | 7.2% | 10.0% |
| equilibrium security h* | 0.86 | 0.99 | 1.10 | 1.15 |
| hacks/yr | 3.15 | 2.80 | 2.45 | 2.10 |

Three results:
1. **The carry trade shrinks monotonically** as alpha capacity tightens
   (C* falls by half; r_pool is arbitraged toward r_market as Prop. E2
   predicts) — but residual collateral demand stays insurance-motivated,
   so collateral/coverage does not collapse to the paper's 11% regime.
2. **Deterrence side effect**: less carry means less posted collateral,
   which weakens forfeiture skin-in-the-game — equilibrium security falls
   (1.15 -> 0.86) and hacks rise 50% (2.10 -> 3.15/yr). The carry trade was
   partly *funding* the deterrent: capital efficiency and security
   incentives trade off.
3. **The core moral-hazard result is robust**: under K=1000, removing
   forfeiture still collapses security (0.99 -> 0.21) and quintuples hacks
   (2.80 -> 10.10/yr).

Equilibrium status (certified by `prove_equilibrium.py`, VERDICT output;
independently red-teamed by an adversarial referee pass in the sense of
Koren, arXiv:2606.22337 — see PROOF_NOTES.md "Provenance"):
- Fact 1 PROVED (exact); Lemma 1 PROVED (EXACT symbolic proof — the
  referee's numerator-coefficient route — plus independent search).
- Solution concept: protocols are AGGREGATE-takers (gamma, pool revenue,
  cap scaling taken as given, like price-taking). The computed paths are
  certified per-epoch epsilon-equilibria of that game (21/24 test epochs
  exact at eps < 1e-9 $M; max eps = $434/quarter). Exact pure equilibrium
  at EVERY epoch is INCONCLUSIVE because it genuinely need not exist in
  the finite-h game (mixed extension covered analytically).
- Distance to FULL-game Nash (deviator internalizes its aggregate
  impact) is MEASURED, not assumed: eps <= $964/quarter for the largest
  collateral posters — economically negligible, but 8 orders above the
  aggregate-taking certificate, which is why the concept is stated
  explicitly (the referee refuted the original "Nash criterion" framing).
- Prop. E2' (corrected): monotone comparative statics in K verified; the
  E3' marginal-yield decomposition matches measurement within 1%; the
  pre-registered "collateral/coverage halves" criterion is not met
  (falls 28%, not 50%) — INCONCLUSIVE as specified, the residual demand
  being insurance-motivated.

## Deliberate scope choices / extensions

- Attacker targets protocol TVL only; the insurance pool itself and posted
  collateral are not attackable (a pool-as-target extension is natural).
- Attack does not shift `V` beyond the realized loss; no reputation dynamics.
- Myopic stage play; forward-looking protocols (deterrence via commitment to
  security) would need value-function iteration.
- One representative attacker population; no attacker capacity constraint
  beyond the opportunity process, no learning.
- Speculator layer is competitive risk-neutral RE pricing (Prop. 1 imposed),
  not an order-book game; insider/manipulation games are out of scope.
- Forfeiture follows the paper (accrues to pool, does not offset claims);
  v6 instead nets collateral against the payout — flagged, not replicated.
