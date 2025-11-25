import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
import math
import streamlit as st

from dashboard_helpers import (
    growth_rate_annualized,
    quantiles_over_time,
    summarize_paths_at_year,
)

# ----------------------------------
# STREAMLIT CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Orbital Debris Risk Outlook",
    layout="wide"
)

st.title("Orbital Debris Risk Outlook (<10 cm)")
st.write("""
Scenario-based Monte Carlo forecasts of large orbital debris and catastrophic collisions,
grounded in SATCAT history and aligned with NASA ODPO / ESA space environment practices to
show how mitigation shifts the future risk curve.
""")
st.caption(
    "NASA Orbital Debris Quarterly News (ODQN 29-3) and ESA Space Environment Report 9.1 inform explosion rates, "
    "mitigation compliance, and long-term outlook assumptions."
)
st.markdown("---")

# ----------------------------------
# SCENARIOS / ORIGINAL CONSTANTS
# ----------------------------------
SCENARIOS = {
    "PMD25_Exp0.0010": {"pmd_comp": 0.90, "P8": 0.0010},
    "PMD25_Exp0.0045": {"pmd_comp": 0.90, "P8": 0.0045},
    "NoMit_Exp0.0010": {"pmd_comp": 0.20, "P8": 0.0010},
    "NoMit_Exp0.0045": {"pmd_comp": 0.20, "P8": 0.0045},
}

SCENARIO_META = {
    "PMD25_Exp0.0010": {
        "label": "PMD 90% | 0.10% expl",
        "color": "#4682B4",
        "pair_key": "Exp0.0010",
        "mitigated": True,
    },
    "PMD25_Exp0.0045": {
        "label": "PMD 90% | 0.45% expl",
        "color": "#2E8B57",
        "pair_key": "Exp0.0045",
        "mitigated": True,
    },
    "NoMit_Exp0.0010": {
        "label": "Status quo | 0.10% expl",
        "color": "#FF7043",
        "pair_key": "Exp0.0010",
        "mitigated": False,
    },
    "NoMit_Exp0.0045": {
        "label": "Status quo | 0.45% expl",
        "color": "#A50021",
        "pair_key": "Exp0.0045",
        "mitigated": False,
    },
}

# ----------------------------------
# USER INPUTS (SIDEBAR)
# ----------------------------------
st.sidebar.header("Scenarios")

# Hard-coded defaults for path and simulation sizing (previous sidebar controls)
DATA_PATH = "./Data"
satcat_filename = "satcat.csv"
HIST_TAIL = 25
N_YEARS_FWD = 200
N_PATHS = 300

selected_scenarios = st.sidebar.multiselect(
    "Scenarios to show",
    options=[
        "PMD25_Exp0.0010",
        "PMD25_Exp0.0045",
        "NoMit_Exp0.0010",
        "NoMit_Exp0.0045",
    ],
    default=[
        "PMD25_Exp0.0010",
        "PMD25_Exp0.0045",
        "NoMit_Exp0.0010",
        "NoMit_Exp0.0045",
    ],
    format_func=lambda x: SCENARIO_META[x]["label"],
)
st.sidebar.expander("What do these scenarios mean?", expanded=False).write(
    "Explosion rates follow NASA measurements (~0.45% historic vs 0.10% mitigated). "
    "Mitigation cases assume PMD (post-mission disposal) at 90% per ESA practice."
)

st.sidebar.header("Financial Assumptions")
COLLISION_COST_MUSD = st.sidebar.slider(
    "Loss per catastrophic collision (USD millions)",
    min_value=50, max_value=1000, value=250, step=25
)
AVOIDANCE_COST_MUSD = st.sidebar.slider(
    "Avoidance/mitigation spend per collision averted (USD millions)",
    min_value=1, max_value=100, value=10, step=1
)

# ----------------------------------
# CONSTANTS / ORIGINAL SCENARIOS
# ----------------------------------
DAY = 365.0
FRAG_EXPLOSION = {"SC": 120, "RB": 260, "SOZ": 160}
FRAG_COLLISION = 500
W_SOZ = 1.0
W_FRAG = 0.02

def find_pair_scenario(name: str) -> str | None:
    """Find the counterpart scenario with the same pair_key but opposite mitigation flag."""
    meta = SCENARIO_META.get(name)
    if not meta:
        return None
    target_mitigated = not meta["mitigated"]
    for scn, scn_meta in SCENARIO_META.items():
        if scn == name:
            continue
        if scn_meta["pair_key"] == meta["pair_key"] and scn_meta["mitigated"] == target_mitigated:
            return scn
    return None


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color (#RRGGBB) to rgba string with given alpha."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    alpha = min(max(alpha, 0.0), 1.0)
    return f"rgba({r},{g},{b},{alpha})"


def format_cost_musd(value_musd: float) -> str:
    """Readable cost string given millions USD."""
    if value_musd >= 1000:
        return f"${value_musd/1000:.2f}B"
    return f"${value_musd:.0f}M"


@st.cache_data(show_spinner=True)
def load_orbital_shell_counts(path: str) -> pd.DataFrame | None:
    """Load orbital shell counts long-form data (Band, Object_Type, Count)."""
    file_path = path.rstrip("/") + "/orbital_shell_counts_long.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    required = {"Band", "Object_Type", "Count"}
    if not required.issubset(df.columns):
        return None
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0)
    df = df.dropna(subset=["Band", "Object_Type"])
    df = df[df["Count"] > 0]
    return df


def prep_orbital_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned df with Group column."""
    obj = df["Object_Type"].astype(str).str.lower()
    is_debris = obj.str.contains("debris") | obj.str.contains("rocket_bodies") | obj.str.contains("r/b")
    out = df.copy()
    out["Group"] = np.where(is_debris, "Debris", "Satellites")
    out["Count"] = pd.to_numeric(out["Count"], errors="coerce").fillna(0)
    out = out[out["Count"] > 0]
    return out


def make_orbital_rings(df: pd.DataFrame, highlight_band: str | None = None):
    """Render orbital band rings (Debris vs Satellites) in the original style."""
    # Config tuned to original figure
    BASE_RADIUS = 1.0
    RING_SPACING = 0.70
    THICKNESS_MIN = 0.25
    THICKNESS_MAX_EXTRA = 1.60
    POWER_EXP = 1.3
    START_ANGLE_DEG = 90.0
    ALPHA = 0.95
    FIGSIZE = (8.5, 8.5)
    MARGIN = 0.60
    COLOR_MAP = {'Debris': '#d62728', 'Satellites': '#1f77b4'}
    BG_COL = '#111219'
    LABEL_FP = FontProperties(family='Arial', weight='bold', size=11)
    TEXT_OUTLINE = [pe.withStroke(linewidth=2.8, foreground='black')]
    CHAR_SPACING_BOOST = 0.10
    CHAR_DEG_AT_R1 = 13.0
    MIN_ARC_DEG = 40
    MAX_ARC_DEG = 120

    def compute_arc_deg(text: str, r: float) -> float:
        n = max(1, len(text))
        deg_per_char = CHAR_DEG_AT_R1 / max(0.2, r)
        raw = (n + 0) * deg_per_char * (1.0 + CHAR_SPACING_BOOST)
        return max(MIN_ARC_DEG, min(MAX_ARC_DEG, raw))

    def curved_label_simple(ax, text, r, theta_center_deg, arc_deg, fp=LABEL_FP,
                            color='white', outline=True):
        text = str(text)
        if not text:
            return
        theta_center = math.radians(theta_center_deg)
        half_span = math.radians(max(5.0, arc_deg) / 2.0) * 0.92
        theta1 = theta_center - half_span
        theta2 = theta_center + half_span

        x1 = r * math.cos(theta1)
        x2 = r * math.cos(theta2)
        if x1 <= x2:
            start, end, step_sign = theta1, theta2, +1.0
        else:
            start, end, step_sign = theta2, theta1, -1.0

        n = len(text)
        if n == 1:
            thetas = [0.5 * (start + end)]
        else:
            usable = abs(end - start)
            base_step = usable / (n - 1)
            step = base_step * (1.0 + CHAR_SPACING_BOOST)
            total = step * (n - 1)
            if total > usable:
                step *= usable / total
            thetas = [start + step_sign * i * step for i in range(n)]

        for ch, ang in zip(text, thetas):
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            rot = math.degrees(ang) - 90.0
            ax.text(
                x, y, ch, ha='center', va='center',
                rotation=rot, rotation_mode='anchor',
                fontproperties=fp, color=color,
                path_effects=(TEXT_OUTLINE if outline else None)
            )

    # Prepare data
    bands = list(df['Band'].drop_duplicates())
    obj = df['Object_Type'].astype(str).str.lower()
    is_debris = obj.str.contains('debris') | obj.str.contains('rocket_bodies') | obj.str.contains('r/b')
    df = df.copy()
    df['Group'] = np.where(is_debris, 'Debris', 'Satellites')
    group_order = ['Debris', 'Satellites']

    agg = df.groupby(['Band', 'Group'], as_index=False)['Count'].sum()
    totals = agg.groupby('Band', as_index=False)['Count'].sum().rename(columns={'Count': 'Total'})
    merged = agg.merge(totals, on='Band', how='left')
    merged['Share'] = np.where(merged['Total'] > 0, merged['Count']/merged['Total'], 0.0)

    band_total_map = {b: float(totals.loc[totals['Band'] == b, 'Total'].values[0]) for b in bands}
    vals = np.array([band_total_map[b] for b in bands], dtype=float)
    vmax = float(vals.max()) if len(vals) else 1.0

    def scale_value(v):
        if vmax <= 0:
            return 0.0
        x = v / vmax
        return x ** POWER_EXP

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw=dict(aspect='equal'), facecolor=BG_COL)
    ax.set_axis_off()

    current_radius = BASE_RADIUS
    max_outer_radius = current_radius
    ring_bounds = []

    for band in bands:
        total = band_total_map.get(band, 0.0)
        if total <= 0:
            continue

        w = float(max(0.0, min(1.0, scale_value(total))))
        thickness = THICKNESS_MIN + THICKNESS_MAX_EXTRA * w
        inner_r, outer_r = current_radius, current_radius + thickness

        bdf = merged[merged['Band'] == band].set_index('Group')
        shares = [float(bdf['Share'].get(g, 0.0)) for g in group_order]

        start = START_ANGLE_DEG % 360.0
        edge_w = 1.0 if band != highlight_band else 2.2
        band_alpha = ALPHA if band != highlight_band else 1.0
        for share, g in zip(shares, group_order):
            sweep = 360.0 * share
            if sweep <= 0:
                continue
            theta1, theta2 = start, (start + sweep) % 360.0
            wedge = Wedge(
                (0, 0), r=outer_r, theta1=theta1, theta2=theta2,
                width=(outer_r - inner_r),
                facecolor=COLOR_MAP.get(g, None),
                edgecolor='white', linewidth=edge_w,
                alpha=band_alpha, label=g
            )
            ax.add_patch(wedge)
            start = (start + sweep) % 360.0

        ring_bounds.append({'band': band, 'inner': inner_r, 'outer': outer_r})
        current_radius = outer_r + RING_SPACING
        max_outer_radius = max(max_outer_radius, outer_r)

    TOP_ANGLE = 90
    OUTER_BLEED = 0.60
    for idx, rb in enumerate(ring_bounds):
        band = rb['band']
        if idx == 0:
            next_inner = ring_bounds[1]['inner'] if len(ring_bounds) > 1 else rb['outer'] + RING_SPACING - 0.2
            r_label = (rb['outer'] + next_inner - 0.2) / 2.0
        elif idx < len(ring_bounds) - 1:
            next_inner = ring_bounds[idx + 1]['inner']
            r_label = (rb['outer'] + next_inner - 0.2) / 2.0
        else:
            r_label = rb['outer'] + OUTER_BLEED * (RING_SPACING - 0.2)

        arc_deg = compute_arc_deg(band, r_label)
        curved_label_simple(ax, band, r=r_label, theta_center_deg=TOP_ANGLE, arc_deg=arc_deg, fp=LABEL_FP)

    R = max_outer_radius + MARGIN
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)

    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l)
            H.append(h)
            L.append(l)
    if H:
        leg = ax.legend(H, L, loc='lower right', frameon=False, title='Object Type')
        plt.setp(leg.get_texts(), fontsize=12, color='white')
        plt.setp(leg.get_title(), fontsize=13, color='white')

    ax.set_title(
        'Distribution of Satellites and Debris Across Orbital Bands',
        pad=20,
        fontsize=18, fontweight='bold', color='white'
    )
    fig.tight_layout()
    return fig


def make_orbital_rings_plotly(df: pd.DataFrame):
    """Interactive sunburst fallback with clearer styling."""
    agg = df.groupby(["Band", "Group"], as_index=False)["Count"].sum()
    agg["Count"] = pd.to_numeric(agg["Count"], errors="coerce").fillna(0)
    fig = px.sunburst(
        agg,
        path=["Band", "Group"],
        values="Count",
        color="Group",
        color_discrete_map={"Debris": "#d62728", "Satellites": "#1f77b4"},
        height=800,
        branchvalues="total",
    )
    fig.update_traces(
        insidetextorientation="radial",
        textinfo="label+percent parent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,.0f}<br>% of parent: %{percentParent:.1%}<extra></extra>",
        marker=dict(line=dict(color="#ffffff", width=1.2)),
    )
    fig.update_layout(
        margin=dict(t=60, l=0, r=0, b=0),
        title="Satellites vs Debris by orbital band (interactive)",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111111", size=14, family="Arial"),
        uniformtext_minsize=12,
        showlegend=True,
    )
    return fig

# ----------------------------------
# LAMBDA SHAPES (FROM YOUR CODE)
# ----------------------------------
def lam_sc_base(days):
    tau = 4.5e4
    return np.exp(-days/tau)

def lam_rb_base(days):
    return (0.70*np.exp(-days/30.0) +
            0.20*np.exp(-days/400.0) +
            0.10*np.exp(-days/5000.0))

def lam_soz_base(days):
    mu, sigma = 10*DAY, 3*DAY
    z = (days - mu)/sigma
    return np.exp(-0.5*z*z)

LAM_BASE = {"SC": lam_sc_base, "RB": lam_rb_base, "SOZ": lam_soz_base}

DECAY_BASE    = 0.009
DECAY_AMP     = 0.20
SOLAR_PERIOD  = 11.0

def annual_decay_factor_core(t_year, base_scale=1.0, amp_scale=1.0):
    return max(
        0.0,
        base_scale * DECAY_BASE * (
            1.0 + amp_scale * DECAY_AMP * np.sin(2*np.pi*(t_year/SOLAR_PERIOD))
        )
    )

def scale_for_P8(class_name, P8_target):
    base = LAM_BASE[class_name]
    t = np.linspace(0.0, 8*DAY, 8001)
    A = np.trapezoid(base(t), t)
    A = max(A, 1e-12)
    return -np.log(1.0 - P8_target)/A

def yearly_hazard_matrix(class_name, k_scale, max_age=200):
    base = LAM_BASE[class_name]
    ages = np.arange(0, max_age+1)
    grid = np.linspace(0, DAY, 721)
    H = np.zeros_like(ages, dtype=float)
    for a in ages:
        tt = a*DAY + grid
        H[a] = k_scale * np.trapezoid(base(tt), tt)
    return H

def amp_rbsoz_from_P8(P8, base=0.001, gain=0.55):
    return 1.0 + gain * (P8/base - 1.0)

def scenario_knobs(pmd_comp: float, P8: float):
    # (your full scenario_knobs function pasted here unchanged)
    k = dict(
        apply_pmd_25 = (pmd_comp >= 0.85),
        frag_decay_mult = 4.0,
        w_frag = W_FRAG,
        exp_frag_amp_RB_SOZ = np.clip((P8/0.001)**0.50, 1.0, 2.5),
        hazard_amp_rbsoz = 1.0,
        decay_base_scale = 1.0,
        decay_amp_scale  = 1.0,
        pmd_scale = 1.0,
        coll_base = 0.05, coll_k = 0.40*(P8/0.001)**0.15, coll_gamma = 0.95,
        coll_cap0 = 0.6, coll_cap_grow = 0.0, coll_cap_pow = 1.6,
        coll_eta_frag = 0.0,
        coll_accel = 0.0,
        coll_alpha = 1.5
    )

    if pmd_comp < 0.5:
        k["apply_pmd_25"] = False
        k["decay_base_scale"] = 0.2
        k["decay_amp_scale"]  = 1.0
        k["frag_decay_mult"]  = 1.0
        k["w_frag"]           = 0.02
        k["hazard_amp_rbsoz"] = amp_rbsoz_from_P8(P8, base=0.001, gain=0.95)
        k["exp_frag_amp_RB_SOZ"] = np.clip((P8/0.001)**0.9, 1.0, 4.0)

        if P8 >= 0.0045:
            k["coll_base"]   = 0.42
            k["coll_k"]      = 1.10
            k["coll_gamma"]  = 1.12
            k["coll_cap0"]   = 4.6
            k["coll_cap_grow"]= 0.35
            k["coll_cap_pow"] = 1.8
            k["coll_eta_frag"] = 0.32
            k["coll_accel"]  = 0.40
            k["coll_alpha"]  = 2.2
        else:
            k["coll_base"]   = 0.24
            k["coll_k"]      = 0.92
            k["coll_gamma"]  = 1.05
            k["coll_cap0"]    = 3.4
            k["coll_cap_grow"]= 0.22
            k["coll_cap_pow"] = 1.6
            k["coll_eta_frag"] = 0.10
            k["coll_accel"]  = 0.22
            k["coll_alpha"]  = 1.8
    else:
        k["w_frag"] = 0.05
        k["frag_decay_mult"] = 2.5
        k["decay_base_scale"] = 0.90
        k["pmd_scale"] = 0.95

        if P8 >= 0.0045:
            k["hazard_amp_rbsoz"] = 1.06
            k["exp_frag_amp_RB_SOZ"] *= 1.05
            k["coll_base"]     = 0.058
            k["coll_k"]        = 0.46
            k["coll_gamma"]    = 0.96
            k["coll_eta_frag"] = 0.015
            k["coll_accel"]    = 0.06
            k["coll_alpha"]    = 1.7
            k["coll_cap0"]     = 0.66
            k["coll_cap_grow"] = 0.12
            k["coll_cap_pow"]  = 1.7
        else:
            k["coll_base"]     = 0.045
            k["coll_k"]        = 0.38
            k["coll_gamma"]    = 0.95
            k["coll_eta_frag"] = 0.0
            k["coll_accel"]    = 0.0
            k["coll_alpha"]    = 1.5
            k["coll_cap0"]     = 0.60
            k["coll_cap_grow"] = 0.05
            k["coll_cap_pow"]  = 1.6

    return k

def collisions_rate(N_eff_parents, N_eff_ref, frag_stock, knobs, t_year, horizon_years):
    base_term = knobs["coll_base"] + knobs["coll_k"] * (max(N_eff_parents,1.0)/max(N_eff_ref,1.0))**knobs["coll_gamma"]
    if knobs.get("coll_eta_frag", 0.0) > 0.0:
        base_term += knobs["coll_eta_frag"] * (frag_stock / max(N_eff_ref,1.0))**1.05

    if knobs.get("coll_accel", 0.0) > 0.0:
        tau = t_year / max(horizon_years, 1)
        alpha = knobs.get("coll_alpha", 1.5)
        ramp = np.expm1(alpha * tau) / np.expm1(alpha)
        base_term *= (1.0 + knobs["coll_accel"] * ramp)

    tau = t_year / max(horizon_years, 1)
    cap_t = knobs["coll_cap0"] * (1.0 + knobs["coll_cap_grow"] * (tau**knobs["coll_cap_pow"]))

    return np.clip(base_term, 0.0, cap_t)

# ----------------------------------
# CACHED LOADING + BASELINE COHORTS
# ----------------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_satcat(path, filename, hist_tail):
    SATCAT_PATH = path.rstrip("/") + "/" + filename

    usecols = ["OBJECT_NAME","OBJECT_TYPE","ORBIT_CENTER","LAUNCH_DATE","DECAY_DATE","OBJECT_ID","NORAD_CAT_ID"]
    sat = pd.read_csv(SATCAT_PATH, usecols=usecols, low_memory=False)
    sat = sat[sat["ORBIT_CENTER"] == "EA"].copy()

    sat["LAUNCH_DATE"] = pd.to_datetime(sat["LAUNCH_DATE"], errors="coerce")
    sat["DECAY_DATE"]  = pd.to_datetime(sat["DECAY_DATE"],  errors="coerce")

    t = sat["OBJECT_TYPE"].astype(str).str.upper()
    name = sat["OBJECT_NAME"].astype(str).str.upper()

    sat["CLASS"] = np.select(
        [
            name.str.contains("AUX MOTOR"),
            t.str.startswith("R/B"),
            t.str.startswith("PAY"),
        ],
        ["SOZ", "RB", "SC"],
        default="EXCLUDE"
    )

    last_year = int(sat["LAUNCH_DATE"].dt.year.dropna().max())
    BASELINE_YEAR = last_year
    cut = pd.Timestamp(year=BASELINE_YEAR, month=1, day=1)
    alive = (sat["LAUNCH_DATE"] <= cut) & (sat["DECAY_DATE"].isna() | (sat["DECAY_DATE"] >= cut))
    active = sat[alive].copy()

    active["AGE_Y"] = ((cut - active["LAUNCH_DATE"]).dt.days // 365).astype("Int64").fillna(0).astype(int)
    active.loc[active["AGE_Y"] < 0, "AGE_Y"] = 0

    classes = ["SC","RB","SOZ"]
    cohorts0 = {}
    max_age = int(active["AGE_Y"].max()) if len(active) else 0
    for c in classes:
        aa = active[active["CLASS"] == c]
        counts = aa.groupby("AGE_Y").size()
        arr = np.zeros(max_age+1, dtype=float)
        if not counts.empty:
            arr[counts.index.values] = counts.values
        cohorts0[c] = arr

    hist = sat.assign(Y=sat["LAUNCH_DATE"].dt.year).dropna(subset=["Y"])
    hist = hist[hist["Y"].between(BASELINE_YEAR - hist_tail, BASELINE_YEAR - 1)]
    launch_by_year = hist[hist["CLASS"].isin(classes)].groupby(["Y","CLASS"]).size().unstack(fill_value=0).reindex(columns=classes, fill_value=0)

    if len(launch_by_year) >= 2:
        tail_mean = launch_by_year.mean(axis=0)
        tail_cagr = (launch_by_year.iloc[-1] / launch_by_year.iloc[0]).pow(1/(len(launch_by_year)-1)) - 1
    else:
        tail_mean = pd.Series({c: 0.0 for c in classes})
        tail_cagr = pd.Series({c: 0.0 for c in classes})

    return BASELINE_YEAR, cohorts0, tail_mean, tail_cagr

BASELINE_YEAR, cohorts0, tail_mean, tail_cagr = load_and_prepare_satcat(
    DATA_PATH, satcat_filename, HIST_TAIL
)

# ----------------------------------
# MONTE CARLO PATHS (CACHED)
# ----------------------------------
def run_one_path(seed, scenario, H, knobs, scale_eff, N_eff_ref, N_years_fwd, tail_mean_local, cohorts0_local):
    rng = np.random.default_rng(seed)

    cohorts = {c: cohorts0_local[c].copy() for c in ["SC","RB","SOZ"]}
    frag_stock = 0.0

    total_eff = np.zeros(N_years_fwd+1)
    cum_coll  = np.zeros_like(total_eff)

    par_sc = cohorts["SC"].sum()
    par_rb = cohorts["RB"].sum()
    par_soz = cohorts["SOZ"].sum()
    N_eff_raw0 = par_sc + par_rb + W_SOZ*par_soz + W_FRAG*frag_stock
    total_eff[0] = N_eff_raw0 * scale_eff

    m_exp = knobs["exp_frag_amp_RB_SOZ"]

    for t in range(1, N_years_fwd+1):
        exp_frags = 0.0
        for c in ["SC","RB","SOZ"]:
            arr, Hc = cohorts[c], H[c]
            Amax = min(len(arr)-1, len(Hc)-1)
            Pyear = 1.0 - np.exp(-Hc[:Amax+1])
            lam = float((arr[:Amax+1] * Pyear).sum())
            parents_now = float(arr[:Amax+1].sum())
            dyn_cap = min(20.0, max(3.0, parents_now / 1000.0))
            lam = np.clip(lam, 0.0, dyn_cap)
            n_ev = rng.poisson(lam)
            per_event = FRAG_EXPLOSION[c] * (m_exp if c in ("RB","SOZ") else 1.0)
            exp_frags += n_ev * per_event

        par_sc = cohorts["SC"].sum()
        par_rb = cohorts["RB"].sum()
        par_soz = cohorts["SOZ"].sum()
        N_eff_parents = par_sc + par_rb + W_SOZ*par_soz
        lam_coll = collisions_rate(N_eff_parents, N_eff_ref, frag_stock, knobs, t_year=t, horizon_years=N_years_fwd)
        n_coll   = rng.poisson(lam_coll)
        coll_frags = n_coll * FRAG_COLLISION

        dr = annual_decay_factor_core(t_year=t, base_scale=knobs["decay_base_scale"], amp_scale=knobs["decay_amp_scale"])
        for c in ["SC","RB","SOZ"]:
            cohorts[c] *= (1.0 - dr)
            cohorts[c][cohorts[c] < 0] = 0.0
        frag_stock *= max(0.0, 1.0 - knobs["frag_decay_mult"]*dr)

        if knobs["apply_pmd_25"]:
            for c in ["SC","RB","SOZ"]:
                if len(cohorts[c]) > 26:
                    r = scenario["pmd_comp"] * knobs.get("pmd_scale", 1.0) * cohorts[c][25]
                    cohorts[c][25] -= r

        for c in ["SC","RB","SOZ"]:
            cohorts[c] = np.concatenate([[0.0], cohorts[c]])

        for c in ["SC","RB","SOZ"]:
            cohorts[c][0] += float(tail_mean_local.get(c, 0.0))

        frag_stock += exp_frags + coll_frags
        par_sc = cohorts["SC"].sum()
        par_rb = cohorts["RB"].sum()
        par_soz = cohorts["SOZ"].sum()
        N_eff_parents = par_sc + par_rb + W_SOZ*par_soz
        N_eff_raw = N_eff_parents + knobs["w_frag"] * frag_stock
        total_eff[t] = N_eff_raw * scale_eff
        cum_coll[t]  = cum_coll[t-1] + n_coll

    return total_eff, cum_coll

@st.cache_data(show_spinner=True)
def run_fan_cached(scn_name, N_paths, N_years_fwd, HIST_TAIL, BASELINE_YEAR, cohorts0_local, tail_mean_local):
    scenario = SCENARIOS[scn_name]
    knobs = scenario_knobs(scenario["pmd_comp"], scenario["P8"])

    k_base = {c: scale_for_P8(c, scenario["P8"]) for c in ["SC","RB","SOZ"]}
    k_scale = {
        "SC":  k_base["SC"],
        "RB":  2.0 * k_base["RB"]  * knobs["hazard_amp_rbsoz"],
        "SOZ": 2.0 * k_base["SOZ"] * knobs["hazard_amp_rbsoz"],
    }
    H = {c: yearly_hazard_matrix(c, k_scale[c], max_age=200) for c in ["SC","RB","SOZ"]}

    N_eff_ref = cohorts0_local["SC"].sum() + cohorts0_local["RB"].sum() + W_SOZ*cohorts0_local["SOZ"].sum()
    N_eff0 = N_eff_ref + W_FRAG*0.0
    TARGET_BASELINE_EFF = 36000.0
    scale_eff = TARGET_BASELINE_EFF / max(N_eff0, 1.0)

    Ns, Cs = [], []
    seed_base = abs(hash((scn_name, HIST_TAIL))) % (2**31)
    for i in range(N_paths):
        n, c = run_one_path(
            seed_base + i, scenario, H, knobs, scale_eff,
            N_eff_ref, N_years_fwd, tail_mean_local, cohorts0_local
        )
        Ns.append(n)
        Cs.append(c)

    Ns = np.vstack(Ns)
    Cs = np.vstack(Cs)
    YEARS = np.arange(BASELINE_YEAR, BASELINE_YEAR + N_years_fwd + 1)
    return YEARS, Ns, Cs

def add_fan_traces(fig, years, paths, meta, show_band=True):
    """Add median + 5-95% band for a scenario."""
    q = quantiles_over_time(paths)
    if show_band:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=q["p95"],
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=q["p05"],
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_hex_to_rgba(meta["color"], 0.14),
                hoverinfo="skip",
                name=None,
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=q["p50"],
            mode="lines",
            line=dict(color=meta["color"], width=3),
            name=meta["label"],
        )
    )

# ----------------------------------
# RUN SIMULATION & PLOT
# ----------------------------------
if not selected_scenarios:
    st.warning("Select at least one scenario in the sidebar to run the simulation.")
else:
    with st.spinner("Running Monte Carlo simulations..."):
        fans = {}
        for scn in selected_scenarios:
            YEARS, Ns, Cs = run_fan_cached(
                scn, N_PATHS, N_YEARS_FWD, HIST_TAIL,
                BASELINE_YEAR, cohorts0, tail_mean
            )
            fans[scn] = {"YEARS": YEARS, "N": Ns, "C": Cs}

    best_scenario = "PMD25_Exp0.0010"
    if best_scenario not in fans:
        YEARS, Ns, Cs = run_fan_cached(
            best_scenario, N_PATHS, N_YEARS_FWD, HIST_TAIL,
            BASELINE_YEAR, cohorts0, tail_mean
        )
        best_pack = {"YEARS": YEARS, "N": Ns, "C": Cs}
    else:
        best_pack = fans[best_scenario]

    worst_scenario = "NoMit_Exp0.0045"
    if worst_scenario not in fans:
        YEARS, Ns, Cs = run_fan_cached(
            worst_scenario, N_PATHS, N_YEARS_FWD, HIST_TAIL,
            BASELINE_YEAR, cohorts0, tail_mean
        )
        worst_pack = {"YEARS": YEARS, "N": Ns, "C": Cs}
    else:
        worst_pack = fans[worst_scenario]

    years = next(iter(fans.values()))["YEARS"]
    default_focus_year = int(min(years[0] + 50, years[-1]))

    st.markdown("### Insight controls")
    focus_cols = st.columns([2, 1])
    with focus_cols[0]:
        focus_scenario = st.selectbox(
            "Scenario spotlight",
            selected_scenarios,
            format_func=lambda x: SCENARIO_META[x]["label"],
        )
    with focus_cols[1]:
        focus_year = st.slider(
            "Year in focus",
            int(years[0]),
            int(years[-1]),
            default_focus_year,
            step=1,
        )

    focus_pack = fans[focus_scenario]
    obj_summary = summarize_paths_at_year(years, focus_pack["N"], focus_year)
    coll_summary = summarize_paths_at_year(years, focus_pack["C"], focus_year)

    base_obj_median = float(np.median(focus_pack["N"][:, 0]))
    elapsed_years = obj_summary["year"] - int(years[0])
    growth = growth_rate_annualized(base_obj_median, obj_summary["median"], elapsed_years)

    best_obj_summary = summarize_paths_at_year(best_pack["YEARS"], best_pack["N"], focus_year)
    best_coll_summary = summarize_paths_at_year(best_pack["YEARS"], best_pack["C"], focus_year)
    best_base_obj = float(np.median(best_pack["N"][:, 0]))
    best_growth = growth_rate_annualized(best_base_obj, best_obj_summary["median"], elapsed_years)

    worst_obj_summary = summarize_paths_at_year(worst_pack["YEARS"], worst_pack["N"], focus_year)
    worst_coll_summary = summarize_paths_at_year(worst_pack["YEARS"], worst_pack["C"], focus_year)

    st.markdown("### Impact snapshot")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric(
        f"{SCENARIO_META[focus_scenario]['label']} | Objects in {obj_summary['year']}",
        f"{obj_summary['median']:,.0f}",
        delta=f"Delta vs best-case: {obj_summary['median'] - best_obj_summary['median']:+,.0f}",
    )
    kpi_cols[1].metric(
        f"Cumulative collisions by {coll_summary['year']}",
        f"{coll_summary['median']:.2f}",
        delta=f"Delta vs best-case: {coll_summary['median'] - best_coll_summary['median']:+.2f}",
    )
    kpi_cols[2].metric(
        "Average annual growth",
        f"{growth*100:.2f}%",
        delta=f"Delta vs best-case: {(growth - best_growth)*100:+.2f}%",
    )
    st.markdown("### Financial & risk framing")
    fin_cols = st.columns(3)
    fin_cols[0].metric(
        "Expected loss (median)",
        format_cost_musd(coll_summary["median"] * COLLISION_COST_MUSD),
        delta=f"Delta vs best-case: {format_cost_musd((coll_summary['median'] - best_coll_summary['median']) * COLLISION_COST_MUSD)}",
    )

    collisions_reduced_vs_worst = worst_coll_summary["median"] - coll_summary["median"]
    reduction_pct_vs_worst = None
    if worst_coll_summary["median"] > 0:
        reduction_pct_vs_worst = collisions_reduced_vs_worst / worst_coll_summary["median"]

    fin_cols[1].metric(
        "Collisions avoided vs worst-case (NoMit 0.45% expl)",
        f"{collisions_reduced_vs_worst:.2f}",
        delta=(f"{reduction_pct_vs_worst*100:+.1f}% vs worst-case" if reduction_pct_vs_worst is not None else None),
    )
    loss_avoided_vs_worst = max(0.0, collisions_reduced_vs_worst) * COLLISION_COST_MUSD
    fin_cols[2].metric(
        "Loss avoided vs worst-case",
        format_cost_musd(loss_avoided_vs_worst),
        delta=f"Worst-case loss: {format_cost_musd(worst_coll_summary['median'] * COLLISION_COST_MUSD)}",
    )

    st.markdown("### Projections")

    obj_fig = go.Figure()
    for name, pack in fans.items():
        add_fan_traces(obj_fig, pack["YEARS"], pack["N"], SCENARIO_META[name])
    obj_fig.update_layout(
        title="Projected Large-Object Population (<10 cm)",
        xaxis_title="Year",
        yaxis_title="Objects (<10 cm, effective count)",
        hovermode="x unified",
        template="simple_white",
        legend_title="Scenarios",
        height=750,
    )
    st.plotly_chart(obj_fig, width="stretch", config={"displayModeBar": False})

    coll_fig = go.Figure()
    for name, pack in fans.items():
        add_fan_traces(coll_fig, pack["YEARS"], pack["C"], SCENARIO_META[name])
    coll_fig.update_layout(
        title="Projected Catastrophic Collisions (cumulative)",
        xaxis_title="Year",
        yaxis_title="Cumulative catastrophic collisions",
        hovermode="x unified",
        template="simple_white",
        legend_title="Scenarios",
        height=750,
    )
    st.plotly_chart(coll_fig, width="stretch", config={"displayModeBar": False})

    st.markdown("### Orbital congestion by band")
    orbital_df = load_orbital_shell_counts(DATA_PATH)
    if orbital_df is None or orbital_df.empty:
        st.info("Orbital shell counts not found. Add `orbital_shell_counts_long.csv` to the Data folder to view the orbital congestion rings.")
    else:
        orbital_df = prep_orbital_bands(orbital_df)
        band_totals = orbital_df.groupby("Band")["Count"].sum().sort_values(ascending=False)
        default_band = band_totals.index[0] if not band_totals.empty else None

        band_choice = st.selectbox(
            "Focus band",
            options=list(band_totals.index),
            index=0 if default_band else None,
        )

        fig_orb = make_orbital_rings_plotly(orbital_df)
        st.plotly_chart(fig_orb, width="stretch", config={"displayModeBar": False})

        # KPI cards for selected band
        band_slice = orbital_df[orbital_df["Band"] == band_choice]
        band_total = float(band_slice["Count"].sum())
        debris_count = float(band_slice.loc[band_slice["Group"] == "Debris", "Count"].sum())
        sat_count = float(band_total - debris_count)
        debris_share = (debris_count / band_total) if band_total > 0 else 0.0
        sat_share = 1.0 - debris_share if band_total > 0 else 0.0

        st.markdown("#### Band snapshot")
        bcols = st.columns(3)
        bcols[0].metric("Total objects", f"{band_total:,.0f}", delta=None)
        bcols[1].metric("Debris share", f"{debris_share*100:.1f}%", delta=f"Debris: {debris_count:,.0f}")
        bcols[2].metric("Satellites share", f"{sat_share*100:.1f}%", delta=f"Satellites: {sat_count:,.0f}")

        # Mini stack comparison vs best/worst
        worst_band = band_totals.index[0]
        best_band = band_totals.index[-1]
        mini = []
        for label, bname in [("Focus", band_choice), ("Most crowded", worst_band), ("Least crowded", best_band)]:
            bdf = orbital_df[orbital_df["Band"] == bname]
            total_b = float(bdf["Count"].sum())
            debris_b = float(bdf.loc[bdf["Group"] == "Debris", "Count"].sum())
            sat_b = total_b - debris_b
            mini.append({"Label": label, "Band": bname, "Type": "Debris", "Count": debris_b})
            mini.append({"Label": label, "Band": bname, "Type": "Satellites", "Count": sat_b})
        mini_df = pd.DataFrame(mini)
        fig_mini = px.bar(
            mini_df,
            x="Label",
            y="Count",
            color="Type",
            color_discrete_map={"Debris": "#d62728", "Satellites": "#1f77b4"},
            barmode="stack",
            title="Band mix comparison",
            height=360,
        )
        fig_mini.update_layout(legend_title="Object Type", xaxis_title=None, yaxis_title="Objects")
        st.plotly_chart(fig_mini, width="stretch", config={"displayModeBar": False})
