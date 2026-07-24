"""Interactive demonstrator for the DeFi insurance dynamic game.

Wraps the ACTUAL research engine in `pipeline/dynamic_game/` (no port, no drift):
a strategic Becker attacker with the FIRM structural cost curve
c(e,V) = (F0 + gamma_c e^kappa) V^eta faces the MARBLE2026 insurance
mechanism; protocols choose collateral AND security anticipating the
attacker; HACK prices are rational-expectations; LP capital follows
Prop. 2(ii).

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud -> this repo -> streamlit_app.py

This app is the DEMONSTRATOR. Canonical, certified results live in
pipeline/dynamic_game/ (run_experiments.py, prove_equilibrium.py,
PROOF_NOTES.md).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "pipeline", "dynamic_game"))
sys.path.insert(0, _HERE)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from params import AllParams
from game import run_game, solve_stage, _aggregates_of
from analysis import summarize, frontier_regression, floor_check
from attacker import participation_threshold
from protocols import init_population, utility_cc, eps_quadrature
import figures
import app_charts

# Self-heal Streamlit's hot-reload path: on a fast redeploy the main
# script is re-read but imported modules stay cached from the previous
# process; reload them if they predate the current code.
if not hasattr(app_charts, "design_heatmap"):
    import importlib
    app_charts = importlib.reload(app_charts)
    figures = importlib.reload(figures)

st.set_page_config(page_title="DeFi Insurance Dynamic Game",
                   page_icon="🛡️", layout="wide")

st.title("DeFi cybersecurity insurance as a dynamic game")
st.caption(
    "A strategic attacker (Becker target selection, FIRM structural cost "
    "estimates) against the market-based insurance mechanism: protocols "
    "choose collateral **and** security anticipating the attacker; HACK "
    "prices are rational-expectations; LP capital self-adjusts. "
    "Exploratory demonstrator — certified results and proofs are in the "
    "repository (`pipeline/dynamic_game/`).")


# ----------------------------- sidebar -----------------------------------
# Attacker-technology parameters are HISTORICAL estimates (structural
# calibration on realized DeFi exploit data, companion paper) — they
# describe the attacker's environment, not design choices. They live under
# an Advanced expander with a reset-to-historical button.
HIST_ATTACKER = dict(w_eta=0.20, w_kappa=2.7, w_F_bar=25.0, w_q_opp=0.04)
for _k, _v in HIST_ATTACKER.items():
    st.session_state.setdefault(_k, _v)


def _reset_attacker():
    for _k, _v in HIST_ATTACKER.items():
        st.session_state[_k] = _v


with st.sidebar:
    st.header("Parameters")
    with st.form("params"):
        st.subheader("Mechanism design (operator's choices)")
        insurance_on = st.toggle("Insurance market on", value=True)
        forfeiture = st.toggle(
            "Collateral forfeiture on hack", value=True,
            help="The deterrence instrument: a hacked protocol's posted "
                 "collateral is transferred to the pool. Switch off to "
                 "see attacker-amplified moral hazard.")
        mu = st.slider(
            "μ — coverage scale", 1.0, 6.0, 3.0, 0.5,
            help="Coverage = μ·C^θ. Scales how much coverage a dollar of "
                 "collateral buys. See the 'Design optimizer' tab before "
                 "hand-tuning.")
        theta = st.slider(
            "θ — coverage concavity", 0.2, 0.9, 0.5, 0.05,
            help="Concavity of coverage in collateral. Low θ caps whale "
                 "coverage harder (capital-efficient, answers η<1 "
                 "whale-targeting); θ→1 approaches proportional coverage.")

        st.subheader("Pool economics")
        r_market_pct = st.slider("r_market (% p.a.)", 1.0, 10.0, 5.0, 0.5)
        r_pool_pct = st.slider("r_pool (% p.a.)", 1.0, 20.0, 10.0, 0.5)
        alpha_mode = st.selectbox(
            "Pool alpha", ["fixed r_pool", "endogenous (scarce alpha)"],
            help="Endogenous: r_pool(C) = r_market + (r_pool0 − r_market)"
                 "·K/(K+C) — excess return decays as capital crowds in")
        K_cap = st.slider("K — alpha capacity ($M, endogenous mode)",
                          100, 8000, 1000, 100)
        CLP0 = st.slider("Initial LP capital ($M)", 50, 1000, 250, 50)

        st.subheader("Simulation")
        n_protocols = st.slider("Protocols", 100, 500, 300, 50)
        n_quarters = st.slider(
            "Quarters", 4, 40, 8, 2,
            help="Simulation horizon. Longer horizons produce more "
                 "incidents (the Frontier tab needs them) but TVL growth "
                 "compounds, so late quarters are larger economies.")
        n_seeds = st.slider(
            "Monte-Carlo seeds", 1, 30, 5,
            help="Independent simulated histories; incidents pool across "
                 "seeds. The cheapest way to give the Frontier tab enough "
                 "hacks: seeds ≥ 15 with quarters ≥ 16 yields ≈ 100+ "
                 "incidents at historical q.")

        with st.expander("Advanced: attacker technology (historical)"):
            st.caption(
                "Estimated from **historical DeFi exploit data** "
                "(structural calibration; companion paper). These describe "
                "the attacker's technology and environment — not the "
                "operator's choices. Vary them for what-if analysis; "
                "**Reset attacker** restores the historical estimates.")
            eta = st.slider(
                "η — value-elasticity of attack cost",
                0.0, 0.5, step=0.01, key="w_eta",
                help="Attack cost ∝ V^η. η=1 would mean doubling a "
                     "protocol's TVL doubles the cost of hacking it (the "
                     "consensus-security benchmark); the historical "
                     "estimate η≈0.20 means cost barely grows with size — "
                     "whales are attractive targets, which is why coverage "
                     "is concave.")
            kappa = st.slider(
                "κ — effort-elasticity of attack cost",
                1.5, 4.0, step=0.1, key="w_kappa",
                help="Attack cost ∝ e^κ within a target: κ≈2.7 "
                     "(historical) means sharply rising marginal cost of "
                     "pushing an exploit deeper — attackers stop at "
                     "partial extraction.")
            F_bar = st.slider(
                "F̄ — attacker outside option ($M)",
                5.0, 100.0, step=5.0, key="w_F_bar",
                help="Participation scale: an opportunity with rent π* is "
                     "executed with probability 1 − exp(−π*/F̄). Low F̄ = "
                     "hungry attackers execute small rents; high F̄ = only "
                     "big scores get taken. Calibrated to historical "
                     "hack frequency.")
            q_opp = st.slider(
                "q — quarterly opportunity probability",
                0.01, 0.20, step=0.01, key="w_q_opp",
                help="Chance per protocol-quarter that an exploitable "
                     "window opens (a vulnerability exists and is found). "
                     "Calibrated to the historical incident rate.")

        submitted = st.form_submit_button("Run simulation", type="primary",
                                          width="stretch")
        st.form_submit_button("Reset attacker to historical values",
                              on_click=_reset_attacker, width="stretch")


def build_cfg():
    r_market = r_market_pct / 100.0
    r_pool = r_pool_pct / 100.0
    endog = alpha_mode.startswith("endogenous")
    if endog and r_pool <= r_market:
        st.warning("Endogenous alpha requires r_pool > r_market; "
                   "using r_pool = r_market + 0.5pp.")
        r_pool = r_market + 0.005
    return dict(eta=eta, kappa=kappa, F_bar=F_bar, q_opp=q_opp,
                insurance_on=insurance_on, forfeiture=forfeiture,
                mu=mu, theta=theta, r_market=r_market, r_pool=r_pool,
                K=(K_cap if endog else None), CLP0=CLP0,
                n_protocols=n_protocols, n_quarters=n_quarters,
                n_seeds=n_seeds)


def params_from(cfg):
    P = AllParams()
    P.attacker.eta = cfg["eta"]
    P.attacker.kappa = cfg["kappa"]
    P.attacker.F_bar = cfg["F_bar"]
    P.attacker.q_opp = cfg["q_opp"]
    P.mech.insurance_on = cfg["insurance_on"]
    P.mech.forfeiture = cfg["forfeiture"]
    P.mech.mu = cfg["mu"]
    P.mech.theta = cfg["theta"]
    P.mech.r_market = cfg["r_market"]
    P.mech.r_pool = cfg["r_pool"]
    P.mech.pool_capacity = cfg["K"]
    P.game.CLP0 = cfg["CLP0"]
    P.proto.n_protocols = cfg["n_protocols"]
    P.game.n_quarters = cfg["n_quarters"]
    return P


@st.cache_data(show_spinner=False, max_entries=24)
def run_scenario(cfg_key: tuple):
    cfg = dict(cfg_key)
    P = params_from(cfg)
    runs = [run_game(P, seed=1000 + s) for s in range(cfg["n_seeds"])]
    summ, ep, inc = summarize(runs, P)
    return summ, ep, inc


@st.cache_data(show_spinner=False, max_entries=96)
def design_point(cfg_key: tuple, n_seeds_opt: int = 3):
    """One (mu, theta) candidate for the design optimizer: small-seed run,
    returns the operator-relevant metrics."""
    cfg = dict(cfg_key)
    P = params_from(cfg)
    runs = [run_game(P, seed=1000 + s) for s in range(n_seeds_opt)]
    summ, ep, _ = summarize(runs, P)
    yrs = cfg["n_quarters"] / 4.0 * n_seeds_opt
    return dict(
        mu=cfg["mu"], theta=cfg["theta"],
        paid_yr=float(ep["paid"].sum()) / yrs,
        shortfall_yr=float(ep["shortfall"].sum()) / yrs,
        hacks_yr=summ["hacks_per_year"], mean_h=summ["mean_h"],
        final_CLP=float(ep.loc[ep["t"] == ep["t"].max(), "CLP"].mean()))


@st.cache_data(show_spinner=False, max_entries=8)
def attacker_facts(cfg_key: tuple):
    cfg = dict(cfg_key)
    P = params_from(cfg)
    vs = participation_threshold(np.array([0.0, 2.0, 4.0]), P.attacker)
    return P, vs


@st.cache_data(show_spinner=False, max_entries=8)
def stage_landscapes(cfg_key: tuple):
    """First-quarter equilibrium snapshot + utility landscapes U(C_C, h)
    for three representative protocol sizes (for the 3-D decision tab)."""
    cfg = dict(cfg_key)
    P = params_from(cfg)
    rng = np.random.default_rng(1000)
    pop = init_population(rng, P)
    eq = solve_stage(pop, P.game.CLP0, P)
    br, atk = eq["br"], eq["atk"]
    agg, _ = _aggregates_of(br["CC"], br["h"], br["p_q"], P.game.CLP0,
                            pop, P)
    eps_n, eps_w = eps_quadrature(P)
    hg = atk["h_grid"]
    out = {}
    for label, tv in [("small (~$10M TVL)", 10.0),
                      ("mid (~$100M TVL)", 100.0),
                      ("whale (~$1B TVL)", 1000.0)]:
        i = int(np.argmin(np.abs(pop["TVL"] - tv)))
        cap = P.proto.cc_tvl_cap * pop["TVL"][i]
        ccg = np.linspace(0.0, cap, 61)
        pop_i = {k: np.repeat(v[i:i + 1], ccg.size) for k, v in pop.items()}
        atk_i = {k: (np.repeat(v[i:i + 1, :], ccg.size, axis=0)
                     if isinstance(v, np.ndarray) and v.ndim == 2 else v)
                 for k, v in atk.items()}
        agg_i = dict(agg, sum_CC_others=np.repeat(
            agg["sum_CC_others"][i:i + 1], ccg.size))
        CC_mat = np.broadcast_to(ccg[:, None], (ccg.size, hg.size)).copy()
        U = utility_cc(CC_mat, pop_i, atk_i, agg_i, P, eps_n, eps_w)
        out[label] = dict(ccg=ccg, hg=hg, U=U, cc_star=float(br["CC"][i]),
                          h_star=float(br["h"][i]),
                          u_star=float(br["utility"][i]),
                          tvl=float(pop["TVL"][i]))
    return out


def show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


cfg = build_cfg()
cfg_key = tuple(sorted(cfg.items()))

with st.spinner(f"Solving the dynamic game ({cfg['n_seeds']} seeds x "
                f"{cfg['n_quarters']} quarters, {cfg['n_protocols']} "
                "protocols)…"):
    summ, ep, inc = run_scenario(cfg_key)

# ----------------------------- headline ----------------------------------
c = st.columns(6)
c[0].metric("Hacks / year", f"{summ['hacks_per_year']:.2f}")
c[1].metric("Median loss", f"${summ['median_loss']:.1f}M"
            if summ["median_loss"] else "—")
c[2].metric("Loss rate", f"{summ['loss_rate_bps']:.0f} bps/yr"
            if summ["loss_rate_bps"] else "—")
c[3].metric("Security h*", f"{summ['mean_h']:.2f}")
c[4].metric("Utilization U", f"{summ['mean_U']:.1f}")
c[5].metric("Pool return", f"{summ['final_r_pool']*100:.2f}%")

tabs = st.tabs(["Dynamics", "Hack ledger", "Moral hazard", "Attacker",
                "Decision landscape (3-D)", "Frontier", "The game",
                "Design optimizer"])

with tabs[0]:
    st.plotly_chart(app_charts.dynamics_chart(ep), width="stretch")
    st.caption(
        f"Mean with 5–95% bands over {cfg['n_seeds']} seeds. Collateral/"
        f"coverage: {summ['collateral_to_coverage']:.2f}"
        if summ["collateral_to_coverage"] else "")

with tabs[1]:
    if len(inc):
        led = inc.sort_values("L", ascending=False)[
            ["seed", "t", "V", "h", "e", "b_realized", "L", "cov",
             "claim", "CC"]].reset_index(drop=True)
        led.index += 1
        a, b_, c_ = st.columns(3)
        a.metric("Incidents", f"{len(led)}")
        b_.metric("Total losses", f"${led.L.sum():,.0f}M")
        c_.metric("Claims paid", f"${led.claim.sum():,.0f}M "
                  f"({led.claim.sum() / max(led.L.sum(), 1e-9):.0%} of losses)")
        st.dataframe(
            led, width="stretch", height=430,
            column_config={
                "seed": st.column_config.NumberColumn("run"),
                "t": st.column_config.NumberColumn("quarter"),
                "V": st.column_config.NumberColumn("TVL", format="$%.1fM"),
                "h": st.column_config.NumberColumn("security h",
                                                   format="%.2f"),
                "e": st.column_config.NumberColumn("effort (actions)",
                                                   format="%.1f"),
                "b_realized": st.column_config.NumberColumn(
                    "blast radius (% of TVL)", format="percent"),
                "L": st.column_config.NumberColumn("loss", format="$%.1fM"),
                "cov": st.column_config.NumberColumn("coverage",
                                                     format="$%.1fM"),
                "claim": st.column_config.NumberColumn("claim paid",
                                                       format="$%.1fM"),
                "CC": st.column_config.NumberColumn("collateral forfeited",
                                                    format="$%.1fM"),
            })
        st.download_button("Download CSV",
                           led.to_csv(index=False).encode(),
                           "hack_ledger.csv", "text/csv")
        st.caption("Every simulated incident across the Monte-Carlo runs, "
                   "largest first. Blast radius = loss / TVL; collateral "
                   "is forfeited to the pool when forfeiture is on.")
    else:
        st.info("No hacks realized under these parameters — raise seeds, "
                "quarters, or the opportunity probability q.")

with tabs[2]:
    st.markdown(
        "Same parameters, three regimes: the full mechanism, coverage "
        "**without** collateral forfeiture, and no insurance. With a "
        "strategic attacker, removing forfeiture collapses equilibrium "
        "security and multiplies the hack rate — the collateral lever is "
        "what stands between the mechanism and attacker-amplified moral "
        "hazard.")
    with st.spinner("Solving the three regimes…"):
        regime_summ = {}
        for name, forf, ins in [("full\nmechanism", True, True),
                                ("no\nforfeiture", False, True),
                                ("no\ninsurance", True, False)]:
            cfg_r = dict(cfg, forfeiture=forf, insurance_on=ins)
            s_r, _, _ = run_scenario(tuple(sorted(cfg_r.items())))
            regime_summ[name.replace("\n", "_")] = s_r
    st.plotly_chart(app_charts.moral_hazard_chart(regime_summ),
                    width="stretch")

with tabs[3]:
    Pf, vstars = attacker_facts(cfg_key)
    a, b_, c_ = st.columns(3)
    a.metric("V*(h=0) — participation threshold", f"${vstars[0]:.1f}M")
    b_.metric("V*(h=2)", f"${vstars[1]:.1f}M")
    c_.metric("V*(h=4)", f"${vstars[2]:.1f}M")
    st.plotly_chart(app_charts.attacker_surface_chart(Pf), width="stretch")
    st.caption(
        "Becker target selection with c(e,V) = (F₀ + γc·e^κ)·V^η. Defense "
        "raises the participation threshold (extensive margin) and lowers "
        "extraction; value raises both effort and attack probability.")
    st.subheader("The attacker-rent landscape (drag to rotate)")
    st.plotly_chart(app_charts.attacker_rent_3d(Pf), width="stretch")
    st.caption(
        "π*(V, h): the attacker's expected rent per target. The flat sea "
        "is the no-attack region; the coastline is the participation "
        "frontier V*(h) — defense pushes the coastline right, value "
        "raises the terrain. Whales are mountains: defense reduces what "
        "is extracted, not whether they are worth attacking.")

with tabs[4]:
    st.markdown(
        "**A protocol's decision problem** U(C_C, h) at the equilibrium "
        "aggregates — the surface protocols climb when they best-respond. "
        "Drag to rotate; the red diamond is the equilibrium choice. "
        "Look for the **carry plateau** (utility rising in collateral up "
        "to the wealth cap when the pool out-earns the market) and the "
        "**security ridge** (Proposition C: forfeiture makes collateral "
        "and security complements).")
    lands = stage_landscapes(cfg_key)
    pick = st.radio("Protocol size", list(lands.keys()), horizontal=True)
    land = lands[pick]
    st.plotly_chart(app_charts.protocol_utility_3d(land), width="stretch")
    st.caption(
        f"Selected protocol TVL: ${land['tvl']:,.0f}M; equilibrium "
        f"C_C = ${land['cc_star']:,.1f}M, h = {land['h_star']:.2f}. "
        "First-quarter equilibrium snapshot; toggle forfeiture or switch "
        "to endogenous alpha in the sidebar and watch the surface reshape.")

with tabs[5]:
    if len(inc) >= 40:
        fr = frontier_regression(inc, tau=0.10)
        fc = floor_check(inc, params_from(cfg))
        a, b_, c_ = st.columns(3)
        a.metric("η̂ (frontier, calibrated: "
                 f"{cfg['eta']:.2f})", f"{fr['eta_hat_relative']:.3f}")
        b_.metric("κ̂ (frontier)", f"{fr['relative']['kappa_hat']:.2f}")
        c_.metric("Incidents above theoretical floor",
                  f"{fc['frac_above_floor']:.0%}")
        st.plotly_chart(app_charts.frontier_chart(inc, fr, params_from(cfg)),
                        width="stretch")
        st.caption(
            "The FIRM two-margin quantile regression re-estimated on the "
            "simulated equilibrium incidents. The downward bias in η̂ "
            "mirrors the empirical paper's lower-bound property.")
    else:
        st.info(
            f"Only {len(inc)} incidents — the frontier regression needs "
            "≥ 40 to be estimable (≈ 100+ for stable estimates). Incidents "
            "pool across Monte-Carlo seeds, so the cheapest fix is more "
            "seeds: try **seeds ≥ 15 with quarters ≥ 16** in the sidebar "
            "(protocols ↑ helps too). Raising q adds hacks but departs "
            "from the historical calibration.")

with tabs[6]:
    show(figures.fig_game_structure())
    st.markdown(
        "**One epoch (quarter):** protocols post collateral $C_C$ and pick "
        "security $h$; speculators quote HACK term structures equal to the "
        "attack hazard the attacker's strategy actually induces; the "
        "mechanism sets coverage, the prudential cap and the yield split; "
        "the attacker selects targets and effort under the participation "
        "constraint; losses realize, claims and forfeitures settle, and "
        "the state transitions.\n\n"
        "**Key results** (details, proofs and certification in the repo): "
        "(1) the collateral-forfeiture lever restores security incentives "
        "to the uninsured benchmark; (2) the pool spread r_pool − r_market "
        "acts as a *security subsidy* routed through forfeiture, with "
        "advantageous selection (Proposition C); (3) compressing pool "
        "alpha kills the carry trade but erodes the deterrent; (4) the "
        "attacker-economics participation frontier re-emerges in "
        "equilibrium and is estimable from simulated incidents.")

with tabs[7]:
    st.subheader("Design optimizer: coverage scale μ and concavity θ")
    st.markdown(
        "Instead of hand-tuning μ and θ, grid-search them holding every "
        "other sidebar setting fixed. **Operator objective (transparent, "
        "illustrative):** maximize *delivered risk transfer* — claims "
        "paid ($M/yr) — subject to **no pool shortfall** and **final LP "
        "capital ≥ 50% of initial** (solvency). The table shows every "
        "candidate so the security/coverage trade-off stays visible.")
    MU_GRID = [1.5, 2.0, 3.0, 4.0, 5.0]
    TH_GRID = [0.30, 0.45, 0.60, 0.75]
    _base = {k: v for k, v in cfg.items()
             if k not in ("mu", "theta", "n_seeds")}
    _base_key = tuple(sorted(_base.items()))
    if not cfg["insurance_on"]:
        st.info("Turn the insurance market on to optimize the design.")
    else:
        stale = st.session_state.get("design_cfg") != _base_key
        if st.button(
                f"Run grid ({len(MU_GRID) * len(TH_GRID)} designs × 3 "
                "seeds — a few minutes on first run)", type="primary") \
                or not stale:
            if stale:
                grid = [(m_, t_) for m_ in MU_GRID for t_ in TH_GRID]
                prog = st.progress(0.0, text="running candidate designs…")
                rows = []
                for gi, (m_, t_) in enumerate(grid):
                    c = dict(_base, mu=m_, theta=t_,
                             n_quarters=min(cfg["n_quarters"], 12))
                    rows.append(design_point(tuple(sorted(c.items()))))
                    prog.progress((gi + 1) / len(grid),
                                  text=f"μ={m_:.1f}, θ={t_:.2f}")
                prog.empty()
                st.session_state.design_df = pd.DataFrame(rows)
                st.session_state.design_cfg = _base_key
            ddf = st.session_state.design_df.copy()
            ddf["feasible"] = ((ddf.shortfall_yr < 1e-6)
                               & (ddf.final_CLP >= 0.5 * cfg["CLP0"]))
            feas = ddf[ddf.feasible]
            if len(feas):
                best = feas.sort_values("paid_yr", ascending=False).iloc[0]
                a, b_, c_, d_ = st.columns(4)
                a.metric("optimal μ*", f"{best.mu:.1f}")
                b_.metric("optimal θ*", f"{best.theta:.2f}")
                c_.metric("risk transfer", f"${best.paid_yr:,.1f}M/yr")
                d_.metric("hacks / year", f"{best.hacks_yr:.2f}")
                st.plotly_chart(
                    app_charts.design_heatmap(ddf, best,
                                              (cfg["mu"], cfg["theta"])),
                    width="stretch")
                st.caption(
                    "Blue cells: delivered risk transfer ($M/yr); ✗ = "
                    "infeasible (pool shortfall or LP capital eroded). "
                    "Star = optimum, circle = your current sidebar "
                    "setting. Grid runs at ≤ 12 quarters for speed.")
            else:
                st.warning("No candidate satisfies the solvency "
                           "constraints under these settings — raise LP "
                           "capital or r_pool.")
            st.dataframe(
                ddf.sort_values(["feasible", "paid_yr"],
                                ascending=[False, False]),
                width="stretch", hide_index=True,
                column_config={
                    "mu": st.column_config.NumberColumn("μ"),
                    "theta": st.column_config.NumberColumn("θ"),
                    "paid_yr": st.column_config.NumberColumn(
                        "claims paid ($M/yr)", format="%.1f"),
                    "shortfall_yr": st.column_config.NumberColumn(
                        "shortfall ($M/yr)", format="%.2f"),
                    "hacks_yr": st.column_config.NumberColumn(
                        "hacks/yr", format="%.2f"),
                    "mean_h": st.column_config.NumberColumn(
                        "security h*", format="%.2f"),
                    "final_CLP": st.column_config.NumberColumn(
                        "final LP capital ($M)", format="%.0f"),
                    "feasible": st.column_config.CheckboxColumn("feasible"),
                })

st.divider()
st.caption(
    "Demonstrator only — indicative, small-sample runs. Canonical "
    "experiments, equilibrium certification (ε-Nash per epoch) and proofs: "
    "`pipeline/dynamic_game/` in this repository.")
