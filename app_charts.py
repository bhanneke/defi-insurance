"""Plotly charts for the Streamlit demonstrator.

Interactive, presentation-grade counterparts of the paper figures in
dynamic_game/figures.py (which stay matplotlib for the manuscripts).
Same validated palette; hover tooltips on every mark; one axis per panel.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREEN, PURPLE, RED = ("#2a78d6", "#eda100", "#1baf7a",
                                    "#4a3aa7", "#e34948")
INK, MUTED, GRID = "#1a1a19", "#52514e", "#e8e7e3"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif",
              size=13, color=INK),
    margin=dict(l=10, r=10, t=70, b=10),
    hovermode="x unified",
    plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                font=dict(size=12)),
)


def _style(fig, title=None, height=460):
    fig.update_layout(**LAYOUT, height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16),
                                     x=0, xanchor="left"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=True,
                     linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    return fig


def _hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _band(fig, ep, col, color, name, row, col_i, unit=""):
    g = ep.groupby("t")[col]
    x = g.mean().index + 1
    m, lo, hi = g.mean().values, g.quantile(0.05).values, \
        g.quantile(0.95).values
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines",
                             line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=row, col=col_i)
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0),
                             fill="tonexty",
                             fillcolor=_hex_rgba(color, 0.14),
                             hoverinfo="skip", showlegend=False),
                  row=row, col=col_i)
    fig.add_trace(go.Scatter(
        x=x, y=m, mode="lines", name=name,
        line=dict(color=color, width=2.5),
        hovertemplate=f"{name}: %{{y:.2f}}{unit}<extra></extra>"),
        row=row, col=col_i)


def dynamics_chart(ep):
    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.09,
        subplot_titles=("Utilization vs. dynamic cap", "LP yield share γ",
                        "Pool capital ($M)", "Cumulative claims ($M)"))
    _band(fig, ep, "U", BLUE, "utilization U", 1, 1)
    _band(fig, ep, "U_cap", ORANGE, "prudential cap", 1, 1)
    _band(fig, ep, "gamma", BLUE, "applied γ (blended)", 1, 2)
    _band(fig, ep, "gamma_raw", ORANGE, "raw γ (Eq. 8)", 1, 2)
    _band(fig, ep, "CLP", BLUE, "LP capital", 2, 1, unit="M")
    _band(fig, ep, "sum_CC", ORANGE, "posted collateral", 2, 1, unit="M")
    ep2 = ep.assign(cum_claims=ep.sort_values("t")
                    .groupby("seed").claims.cumsum())
    _band(fig, ep2, "cum_claims", BLUE, "cumulative claims", 2, 2, unit="M")
    fig.update_yaxes(range=[0, 1], row=1, col=2)
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(title_text="quarter", row=r, col=c,
                         title_standoff=4)
    _style(fig, height=560)
    fig.update_annotations(font_size=13)
    return fig


def moral_hazard_chart(summaries):
    regimes = list(summaries.keys())
    labels = [r.replace("_", " ") for r in regimes]
    metrics = [("mean_h", "Equilibrium security h*", ""),
               ("hacks_per_year", "Hacks per year", ""),
               ("raw_losses_per_year", "Raw hack losses", " $M/yr")]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=[m[1] for m in metrics])
    for i, (key, _, unit) in enumerate(metrics, start=1):
        vals = [summaries[r][key] for r in regimes]
        fig.add_trace(go.Bar(
            x=labels, y=vals, marker_color=BLUE, width=0.55,
            text=[f"{v:,.2f}" for v in vals], textposition="outside",
            textfont=dict(size=12, color=INK), cliponaxis=False,
            hovertemplate="%{x}: %{y:,.2f}" + unit + "<extra></extra>",
            showlegend=False), row=1, col=i)
    _style(fig, height=360)
    fig.update_layout(hovermode="closest")
    fig.update_annotations(font_size=13)
    return fig


def attacker_surface_chart(P, vstar_grid=None):
    from attacker import attacker_best_response, participation_threshold
    V = np.logspace(0, 4, 120)
    h = np.linspace(0, 4, 90)
    Vm, Hm = np.meshgrid(V, h, indexing="ij")
    br = attacker_best_response(Vm.ravel(), Hm.ravel(), P.attacker)
    sel = (br["pi_star"] > 0).reshape(Vm.shape)
    e = np.where(sel, br["e_star"].reshape(Vm.shape), np.nan)
    b = np.where(sel, br["b_star"].reshape(Vm.shape), np.nan)
    p1 = (1 - (1 - br["p_q"]) ** 4).reshape(Vm.shape) * 100
    vstar = participation_threshold(h, P.attacker)
    lx = np.log10(V)

    titles = ("optimal effort e* (actions)", "expected blast radius",
              "annualized attack probability (%)")
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                        subplot_titles=titles)
    surfaces = [(e, "%{z:.1f} actions", 0.288), (b, "%{z:.2f}", 0.645),
                (p1, "%{z:.1f}%", 1.0)]
    for i, (Z, ztpl, cb_x) in enumerate(surfaces, start=1):
        fig.add_trace(go.Heatmap(
            x=lx, y=h, z=Z.T, colorscale="Blues", showscale=True,
            colorbar=dict(x=cb_x, len=0.82, thickness=12, y=0.42),
            hovertemplate=("V = $%{customdata:,.0f}M, h = %{y:.1f}<br>"
                           + ztpl + "<extra></extra>"),
            customdata=np.broadcast_to(V, Z.T.shape)), row=1, col=i)
        fig.add_trace(go.Scatter(
            x=np.log10(vstar), y=h, mode="lines",
            line=dict(color=INK, width=2, dash="dash"),
            name="participation threshold V*(h)",
            showlegend=(i == 1),
            hovertemplate="V*(h=%{y:.1f}) = $%{customdata:,.0f}M"
                          "<extra></extra>",
            customdata=vstar), row=1, col=i)
        fig.update_xaxes(tickvals=[0, 1, 2, 3, 4],
                         ticktext=["$1M", "$10M", "$100M", "$1B", "$10B"],
                         title_text="value at stake (log)", row=1, col=i,
                         title_standoff=4)
    fig.update_yaxes(title_text="security level h", row=1, col=1)
    _style(fig, height=380)
    fig.update_layout(hovermode="closest")
    fig.update_annotations(font_size=13)
    return fig


def frontier_chart(inc, frontier, P):
    from attacker import attack_cost
    Vmed = float(inc.V.median())
    rel = frontier["relative"]
    b_adj = np.exp(np.log(inc.b_realized)
                   - rel["v_elast"] * (np.log(inc.V) - np.log(Vmed)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inc.e, y=b_adj, mode="markers", name="simulated incidents",
        marker=dict(color=_hex_rgba(BLUE, 0.55), size=8,
                    line=dict(width=0)),
        customdata=np.stack([inc.V, inc.L], axis=1),
        hovertemplate=("effort %{x:.1f} actions, b = %{y:.3f}<br>"
                       "V = $%{customdata[0]:,.0f}M, "
                       "loss $%{customdata[1]:,.1f}M<extra></extra>")))
    eg = np.linspace(float(inc.e.min()), float(inc.e.max()), 80)
    bhat = np.exp(rel["const"] + rel["kappa_hat"] * np.log(eg)
                  + rel["v_elast"] * np.log(Vmed))
    fig.add_trace(go.Scatter(
        x=eg, y=bhat, mode="lines", name="estimated τ = 0.10 frontier",
        line=dict(color=ORANGE, width=3), hoverinfo="skip"))
    floor = attack_cost(eg, Vmed, P.attacker) / Vmed
    fig.add_trace(go.Scatter(
        x=eg, y=floor, mode="lines",
        name="theoretical floor c(e,V)/V",
        line=dict(color=INK, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(type="log", title_text="attack effort (actions)")
    fig.update_yaxes(type="log",
                     title_text="blast radius L/V (value-adjusted)")
    _style(fig, height=460)
    fig.update_layout(hovermode="closest")
    return fig
