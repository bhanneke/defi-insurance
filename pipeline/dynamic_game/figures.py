"""Figures for the dynamic-game experiments.

Palette: validated categorical slots (dataviz skill reference instance),
fixed order blue, orange, green, purple, red; red and green never adjacent.
One axis per panel; recessive grid; direct labels where possible.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, PURPLE, RED = ("#2a78d6", "#eda100", "#1baf7a",
                                    "#4a3aa7", "#e34948")
INK, MUTED, GRID = "#1a1a19", "#52514e", "#e5e4e0"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _finish(fig, path):
    """Save-and-close when a path is given; return the figure otherwise
    (interactive/Streamlit use)."""
    if path is None:
        return fig
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return None


def _band(ax, ep, col, color, label):
    g = ep.groupby("t")[col]
    m, lo, hi = g.mean(), g.quantile(0.05), g.quantile(0.95)
    ax.plot(m.index + 1, m.values, color=color, lw=2, label=label)
    ax.fill_between(m.index + 1, lo.values, hi.values, color=color, alpha=0.15,
                    lw=0)


def fig_dynamics(ep, path=None):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6))
    ax = axes[0, 0]
    _band(ax, ep, "U", BLUE, "utilization $U$")
    _band(ax, ep, "U_cap", ORANGE, "prudential cap $U_{max}$")
    ax.set_title("Utilization vs. dynamic cap")
    ax.set_xlabel("quarter"); ax.legend(frameon=False)

    ax = axes[0, 1]
    _band(ax, ep, "gamma", BLUE, r"applied $\gamma_t$ (blended)")
    _band(ax, ep, "gamma_raw", ORANGE, r"raw $\gamma$ (Eq. 8)")
    ax.set_title("LP yield share"); ax.set_xlabel("quarter")
    ax.set_ylim(0, 1); ax.legend(frameon=False)

    ax = axes[1, 0]
    _band(ax, ep, "CLP", BLUE, "LP capital $C_{LP}$")
    _band(ax, ep, "sum_CC", ORANGE, "posted collateral $\\Sigma C_C$")
    ax.set_title("Pool capital (\\$M)"); ax.set_xlabel("quarter")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    cum = ep.sort_values("t").groupby("seed").claims.cumsum()
    ep2 = ep.assign(cum_claims=cum)
    _band(ax, ep2, "cum_claims", BLUE, "cumulative claims (\\$M)")
    ax.set_title("Cumulative claims (\\$M)"); ax.set_xlabel("quarter")
    fig.suptitle("Endogenous-attacker dynamic game: baseline dynamics "
                 "(mean, 5–95% across seeds)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, path)


def fig_moral_hazard(summaries, path=None):
    """summaries: dict regime -> summary dict."""
    regimes = list(summaries.keys())
    metrics = [("mean_h", "Equilibrium security $h^*$"),
               ("hacks_per_year", "Hacks per year"),
               ("raw_losses_per_year", "Raw hack losses (\\$M/yr)")]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    x = np.arange(len(regimes))
    for ax, (key, title) in zip(axes, metrics):
        vals = [summaries[r][key] for r in regimes]
        ax.bar(x, vals, width=0.55, color=BLUE, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f" {v:.2f}", ha="center", va="bottom", fontsize=8,
                    color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Moral hazard and the collateral lever (strategic attacker)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, path)


def fig_eta_sweep(decile_profiles, loss_rates, path=None):
    """decile_profiles: dict eta -> (deciles 1..10, annual attack prob).
    loss_rates: dict eta -> loss rate bps."""
    # order low->high eta; colors keep red/green non-adjacent
    colors = [BLUE, ORANGE, GREEN, PURPLE, RED]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4),
                                  gridspec_kw={"width_ratios": [3, 2]})
    items = sorted(decile_profiles.items())
    ends = np.array([100 * prof[1][-1] for _, prof in items])
    # spread end-of-line labels so they never collide
    order = np.argsort(ends)
    ymax = ends.max()
    spread = ends.copy()
    min_gap = 0.055 * ymax
    for k in range(1, len(order)):
        lo_i, hi_i = order[k - 1], order[k]
        if spread[hi_i] - spread[lo_i] < min_gap:
            spread[hi_i] = spread[lo_i] + min_gap
    for c, (eta, prof), y_lab in zip(colors, items, spread):
        d, p = prof
        ax.plot(d, 100 * p, color=c, lw=2)
        ax.annotate(f"$\\eta$={eta:g}", xy=(10, y_lab),
                    xytext=(10.15, y_lab), textcoords="data",
                    color=c, fontsize=8, va="center")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("TVL decile (1 = smallest)")
    ax.set_ylabel("annualized attack probability (%)")
    ax.set_title("Who gets attacked: value-elasticity of attack cost")
    ax.set_xlim(0.8, 11.6)
    ax.set_ylim(top=1.12 * float(spread.max()))

    etas = sorted(loss_rates.keys())
    ax2.plot(etas, [loss_rates[e] for e in etas], color=BLUE, lw=2,
             marker="o", ms=5)
    ax2.axvline(0.20, color=MUTED, lw=1, ls="--")
    ax2.annotate("FIRM estimate\n$\\hat\\eta \\approx 0.20$", xy=(0.20, 0),
                 xytext=(0.23, 0.75), textcoords=("data", "axes fraction"),
                 fontsize=8, color=MUTED)
    ax2.set_xlabel(r"value-elasticity of attack cost $\eta$")
    ax2.set_ylabel("loss rate (bps of coverage/yr)")
    ax2.set_title("System loss rate vs. $\\eta$")
    fig.tight_layout()
    return _finish(fig, path)


def fig_game_structure(path=None):
    """Swimlane timing diagram of one epoch (quarter) of the dynamic game."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    ax.set_xlim(-6, 104); ax.set_ylim(0, 104)
    ax.axis("off"); ax.grid(False)

    lanes = [("Speculators", 82, GREEN), ("Protocols", 63, BLUE),
             ("Pool /\noperator", 44, PURPLE), ("LPs", 25, ORANGE),
             ("Attacker", 6, RED)]
    for name, y, col in lanes:
        ax.text(-5.5, y + 6.5, name, fontsize=9, fontweight="bold",
                color=col, va="center", ha="left")
        ax.axhline(y - 1.5, xmin=0.0, xmax=1.0, color=GRID, lw=0.7)

    stages = [("1 · choices", 8), ("2 · mechanism", 32), ("3 · attacker", 56),
              ("4 · settlement", 80)]
    for name, x in stages:
        ax.text(x + 9, 101.5, name, fontsize=8.5, fontweight="bold",
                color=MUTED, ha="center")

    def box(x, y, text, col, w=19, h=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.6", linewidth=1.1,
                     edgecolor=col, facecolor="white", zorder=3))
        ax.text(x + w / 2, y + h / 2, text, fontsize=7.1, ha="center",
                va="center", color=INK, zorder=4)

    def arrow(x0, y0, x1, y1, col, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                     mutation_scale=11, lw=1.1, color=col, linestyle=ls,
                     zorder=2, shrinkA=2, shrinkB=2))

    # Stage 1: simultaneous choices
    box(8, 82, "quote HACK/NOHACK\nterm structure\n(RE: price = induced hazard)", GREEN)
    box(8, 63, "choose collateral $C_C$\nand security $h$\n(anticipating attacker)", BLUE)
    box(8, 25, "risk capital $C_{LP}$\n(Prop. 2(ii) adjustment)", ORANGE)
    # Stage 2: mechanism aggregates
    box(32, 44, "coverage $\\mu C_C^{\\theta}(1{+}\\xi(h))$\n"
        "cap $U \\leq U_{max}(\\hat p_{1Y})$\n"
        "yield share $\\gamma(U, P_{risk})$\npool return $r_{pool}(C)$", PURPLE, h=14)
    # Stage 3: attacker best response
    box(53, 4, "observes $(V, h)$\n$e^{*} =\\arg\\max\\; \\bar\\epsilon\\,\\beta(e,h)V - c(e,V)$\n"
        "$c(e,V) = (F_0 + \\gamma_c e^{\\kappa})V^{\\eta}$\nattacks iff $\\pi^{*} > F$", RED, w=22, h=15)
    # Stage 4: settlement
    box(80, 82, "trading fees accrue\nto the pool; tokens settle", GREEN)
    box(80, 63, "receive $\\min(cov, L)$\nforfeit $C_C$ if hacked\n$+(1{-}\\gamma)$ yield share", BLUE)
    box(80, 25, "pay claims from $C_{LP}$\n$+\\gamma$ yield share\n$+$ forfeited collateral", ORANGE)
    box(80, 4, "extracts\n$L = \\epsilon\\,\\beta(e^{*}\\!,h)\\,V$", RED, w=16)

    # Flows
    arrow(27, 69, 33, 55, BLUE)       # collateral -> pool
    arrow(27, 30, 33, 45, ORANGE)     # LP capital -> pool
    arrow(27, 86, 41, 59, GREEN)      # prices -> gamma
    arrow(18, 62, 56, 13, BLUE, ls="--")   # (V,h) observed by attacker
    arrow(88, 18, 88, 24, RED)        # losses -> claims
    arrow(85, 62, 87, 38, BLUE, ls="--")   # forfeiture -> LPs/pool

    # within-quarter equilibrium bracket
    ax.plot([7, 7, 80, 80], [97.5, 99, 99, 97.5], color=MUTED, lw=1.1)
    ax.text(43, 96, "within-quarter stage equilibrium: fixed point + "
            "per-epoch $\\varepsilon$-Nash certificate",
            fontsize=8, color=MUTED, ha="center", style="italic")
    # transition
    ax.annotate("", xy=(103, 50), xytext=(100.5, 50),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(102.2, 57, "state transition\n$TVL_{t+1} = TVL_t(1{+}g) - L$\n"
            "$C_{LP}$: Prop. 2(ii)\n$\\to$ quarter $t{+}1$",
            fontsize=7.2, color=INK, ha="right", va="center")

    ax.set_title("One epoch of the dynamic game: players, timing, and flows",
                 fontsize=11, fontweight="bold", pad=16)
    fig.tight_layout()
    return _finish(fig, path)


def fig_attacker_surface(P, path=None):
    """Attacker best-response surfaces over the (V, h) plane, with the
    participation threshold V*(h) overlaid. Sequential single hue."""
    from attacker import attacker_best_response, participation_threshold
    V = np.logspace(0, 4, 160)
    h = np.linspace(0, 4, 120)
    Vm, Hm = np.meshgrid(V, h, indexing="ij")
    br = attacker_best_response(Vm.ravel(), Hm.ravel(), P.attacker)
    e = br["e_star"].reshape(Vm.shape)
    b = br["b_star"].reshape(Vm.shape)
    p1 = (1 - (1 - br["p_q"]) ** 4).reshape(Vm.shape) * 100
    sel = (br["pi_star"] > 0).reshape(Vm.shape)
    e = np.where(sel, e, np.nan)
    b = np.where(sel, b, np.nan)
    vstar = participation_threshold(h, P.attacker)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    specs = [(e, "optimal effort $e^*$ (actions)", axes[0]),
             (b, "expected blast radius $\\beta(e^*,h)$", axes[1]),
             (p1, "annualized attack probability (%)", axes[2])]
    for Z, title, ax in specs:
        pc = ax.pcolormesh(V, h, Z.T, cmap="Blues", shading="auto",
                           rasterized=True)
        ax.plot(vstar, h, color=INK, lw=1.6, ls="--")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("value at stake $V$ (\\$M, log)")
        ax.grid(False)
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("security level $h$")
    axes[0].annotate("$V^*(h)$: participation\nthreshold (Prop. 1)",
                     xy=(vstar[60], h[60]), xytext=(1.6, 3.4),
                     fontsize=8, color=INK,
                     arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    fig.suptitle("Attacker best response to value at stake and defense "
                 f"($\\eta$={P.attacker.eta}, $\\kappa$={P.attacker.kappa})",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, path)


def fig_pool_alpha(panels, path=None):
    """Endogenous-alpha extension: sequential single-hue ramp over the
    ordered capacity K (light = tight capacity, dark = fixed alpha)."""
    ramp = {"250": "#9ec4f0", "1000": "#5f9fe8", "4000": "#2a78d6",
            "inf": "#14477e"}
    label = {"250": "K=250", "1000": "K=1000", "4000": "K=4000",
             "inf": "fixed $r_{pool}$ (K=$\\infty$)"}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    specs = [("r_pool", "Pool return $r_{pool}(C)$ (annual)", axes[0, 0]),
             ("cc_to_cov", "Collateral / coverage", axes[0, 1]),
             ("mean_h", "Equilibrium security $h^*$", axes[1, 0])]
    for key, title, ax in specs:
        for k in ["250", "1000", "4000", "inf"]:
            ax.plot(panels[k]["t"], panels[k][key], color=ramp[k], lw=2)
        ax.set_title(title)
        ax.set_xlabel("quarter")
    axes[0, 0].axhline(0.05, color=MUTED, lw=1, ls="--")
    axes[0, 0].annotate("$r_{market}$", xy=(1, 0.05), xytext=(0, 4),
                        textcoords="offset points", fontsize=8, color=MUTED)
    # shared legend via direct labels on the last line panel
    for k in ["250", "1000", "4000", "inf"]:
        axes[0, 1].annotate(label[k],
                            xy=(panels[k]["t"][-1], panels[k]["cc_to_cov"][-1]),
                            xytext=(4, 0), textcoords="offset points",
                            color=ramp[k], fontsize=8, va="center")
    axes[0, 1].set_xlim(right=panels["inf"]["t"][-1] + 2.6)

    ax = axes[1, 1]
    ks = ["250", "1000", "4000", "inf"]
    x = np.arange(len(ks))
    vals = [panels[k]["hacks_yr"] for k in ks]
    ax.bar(x, vals, width=0.55, color=[ramp[k] for k in ks], zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v, f" {v:.2f}", ha="center", va="bottom", fontsize=8,
                color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([label[k].replace(" (K=$\\infty$)", "") for k in ks],
                       fontsize=8)
    ax.set_title("Hacks per year")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Endogenous pool alpha: the carry trade shrinks — and so "
                 "does the forfeiture deterrent", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, path)


def fig_frontier(inc, frontier, P, path=None):
    """Value-adjusted frontier: effort and value are correlated in
    equilibrium, so the display residualizes the estimated value slope
    (points shifted to the median V) before drawing the tau=0.10 fit and
    the calibrated theoretical floor at median V."""
    from attacker import attack_cost
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Vmed = float(inc.V.median())
    rel = frontier["relative"]
    b_adj = np.exp(np.log(inc.b_realized)
                   - rel["v_elast"] * (np.log(inc.V) - np.log(Vmed)))
    ax.scatter(inc.e, b_adj, s=14, color=BLUE, alpha=0.45, lw=0,
               zorder=3, label="simulated incidents (value-adjusted)")
    eg = np.linspace(inc.e.min(), inc.e.max(), 100)
    bhat = np.exp(rel["const"] + rel["kappa_hat"] * np.log(eg)
                  + rel["v_elast"] * np.log(Vmed))
    ax.plot(eg, bhat, color=ORANGE, lw=2,
            label=r"estimated $\tau=0.10$ frontier")
    floor = attack_cost(eg, Vmed, P.attacker) / Vmed
    ax.plot(eg, floor, color=INK, lw=1.4, ls="--",
            label="theoretical floor $c(e,V)/V$ at median $V$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("attack effort $e$ (actions)")
    ax.set_ylabel("blast radius $b = L/V$, adjusted to median $V$")
    ax.set_title("The participation frontier re-emerges in the simulated "
                 "equilibrium")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _finish(fig, path)


def fig_decoupling(df, path=None):
    """Proposition D: the indemnity-only coverage-security frontier and the
    forfeiture mechanism outside it. df rows: one per mechanism point with
    columns [label, d, family, share_contract, mean_h, hacks_per_year]."""
    fam = df[df["family"] == "indemnity"].sort_values("share_contract")
    forf = df[df["family"] == "forfeiture"].iloc[0]
    noins = df[df["family"] == "none"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, col, ylab in ((axes[0], "mean_h", "equilibrium security $h^*$"),
                          (axes[1], "hacks_per_year", "hacks per year")):
        ax.plot(fam["share_contract"] * 100, fam[col], color=ORANGE, lw=2,
                marker="o", ms=4.5, zorder=3,
                label="indemnity-only contracts (sweep $d$)")
        for _, r in fam.iterrows():
            ax.annotate(f"$d$={r['d']:.2g}",
                        (r["share_contract"] * 100, r[col]),
                        textcoords="offset points", xytext=(4, -10),
                        fontsize=7, color=MUTED)
        ax.scatter([noins["share_contract"] * 100], [noins[col]],
                   marker="s", s=45, color=MUTED, zorder=4,
                   label="no insurance")
        ax.scatter([forf["share_contract"] * 100], [forf[col]],
                   marker="*", s=190, color=BLUE, zorder=5,
                   label="forfeiture mechanism")
        ax.set_xlabel("indemnified share of losses (%)")
        ax.set_ylabel(ylab)
    axes[0].set_title("One instrument, two targets: the coverage–"
                      "security trade-off…")
    axes[1].set_title("…and the attacker feedback that amplifies it")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    return _finish(fig, path)
