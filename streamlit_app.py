"""Interactive demonstrator for the DeFi insurance dynamic game.

Wraps the ACTUAL research engine in `dynamic_game/` (no port, no drift):
a strategic Becker attacker with the FIRM structural cost curve
c(e,V) = (F0 + gamma_c e^kappa) V^eta faces the MARBLE2026 insurance
mechanism; protocols choose collateral AND security anticipating the
attacker; HACK prices are rational-expectations; LP capital follows
Prop. 2(ii).

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud -> this repo -> streamlit_app.py

This app is the DEMONSTRATOR. Canonical, certified results live in
dynamic_game/ (run_experiments.py, prove_equilibrium.py, PROOF_NOTES.md).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "dynamic_game"))
sys.path.insert(0, _HERE)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from params import AllParams
from game import run_game
from analysis import summarize, frontier_regression, floor_check
from attacker import participation_threshold
import figures
import app_charts

st.set_page_config(page_title="DeFi Insurance Dynamic Game",
                   page_icon="🛡️", layout="wide")

st.title("DeFi cybersecurity insurance as a dynamic game")
st.caption(
    "A strategic attacker (Becker target selection, FIRM structural cost "
    "estimates) against the market-based insurance mechanism: protocols "
    "choose collateral **and** security anticipating the attacker; HACK "
    "prices are rational-expectations; LP capital self-adjusts. "
    "Exploratory demonstrator — certified results and proofs are in the "
    "repository (`dynamic_game/`).")


# ----------------------------- sidebar -----------------------------------
with st.sidebar:
    st.header("Parameters")
    with st.form("params"):
        st.subheader("Attacker (FIRM estimates)")
        eta = st.slider("η — value-elasticity of attack cost",
                        0.0, 0.5, 0.20, 0.01,
                        help="Structural estimate: 0.20–0.29 (frontier ~0.20)")
        kappa = st.slider("κ — effort-elasticity of attack cost",
                          1.5, 4.0, 2.7, 0.1)
        F_bar = st.slider("F̄ — attacker outside option ($M)",
                          5.0, 100.0, 25.0, 5.0)
        q_opp = st.slider("q — quarterly opportunity probability",
                          0.01, 0.20, 0.04, 0.01)

        st.subheader("Mechanism")
        insurance_on = st.toggle("Insurance market on", value=True)
        forfeiture = st.toggle("Collateral forfeiture on hack", value=True)
        mu = st.slider("μ — coverage scale", 1.0, 6.0, 3.0, 0.5)
        theta = st.slider("θ — coverage concavity", 0.2, 0.9, 0.5, 0.05)

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
        n_quarters = st.slider("Quarters", 4, 16, 8, 2)
        n_seeds = st.slider("Monte-Carlo seeds", 1, 15, 5)

        submitted = st.form_submit_button("Run simulation", type="primary",
                                          width="stretch")


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


@st.cache_data(show_spinner=False, max_entries=8)
def attacker_facts(cfg_key: tuple):
    cfg = dict(cfg_key)
    P = params_from(cfg)
    vs = participation_threshold(np.array([0.0, 2.0, 4.0]), P.attacker)
    return P, vs


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

tabs = st.tabs(["Dynamics", "Moral hazard", "Attacker", "Frontier",
                "The game"])

with tabs[0]:
    st.plotly_chart(app_charts.dynamics_chart(ep), width="stretch")
    st.caption(
        f"Mean with 5–95% bands over {cfg['n_seeds']} seeds. Collateral/"
        f"coverage: {summ['collateral_to_coverage']:.2f}"
        if summ["collateral_to_coverage"] else "")

with tabs[1]:
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

with tabs[2]:
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

with tabs[3]:
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
        st.info(f"Only {len(inc)} incidents — the frontier regression "
                "needs ≥ 40. Raise the seeds, quarters, or the "
                "opportunity probability q.")

with tabs[4]:
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

st.divider()
st.caption(
    "Demonstrator only — indicative, small-sample runs. Canonical "
    "experiments, equilibrium certification (ε-Nash per epoch) and proofs: "
    "`dynamic_game/` in this repository.")
