import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from dashboard_helpers import (
    format_musd,
    format_pct,
    pct_delta,
    quantiles_over_time,
    summarize_paths_at_year,
)

# ----------------------------------
# STREAMLIT CONFIG & THEME
# ----------------------------------
APP_THEME = {
    "primary": "#f5bf3c",   # orbital gold
    "secondary": "#6ac8ff", # accent cyan
    "amber": "#f29f05",
    "scarlet": "#ff5c5c",
    "surface": "#0f1626",
    "ink": "#f7f9fd",
    "muted": "#9fb0c8",
}

SENTIMENT_COLORS = {
    "good": "#15803d",
    "neutral": "#d97706",
    "bad": "#b91c1c",
}


def _generate_star_positions(
    n: int,
    rng: np.random.Generator,
    cluster_fraction: float = 0.25,
    n_clusters: int = 4,
    cluster_spread: float = 6.0,
) -> list[tuple[float, float]]:
    """Stratified jittered grid plus a few clustered points for even-yet-random coverage."""
    positions: list[tuple[float, float]] = []
    base_count = max(int(n * (1 - cluster_fraction)), 0)

    # Stratified grid for even coverage
    grid_side = max(int(np.ceil(np.sqrt(base_count))), 1)
    cell = 100.0 / grid_side
    for row in range(grid_side):
        for col in range(grid_side):
            if len(positions) >= base_count:
                break
            jitter_x = rng.uniform(-0.35, 0.35) * cell
            jitter_y = rng.uniform(-0.35, 0.35) * cell
            x = (col + 0.5) * cell + jitter_x
            y = (row + 0.5) * cell + jitter_y
            positions.append((np.clip(x, 0, 100), np.clip(y, 0, 100)))

    # Clustered points for natural variation
    cluster_count = n - len(positions)
    if cluster_count > 0:
        centers = rng.uniform(12, 88, size=(n_clusters, 2))
        counts = rng.multinomial(cluster_count, [1 / n_clusters] * n_clusters)
        for idx, c in enumerate(counts):
            if c == 0:
                continue
            cx, cy = centers[idx]
            xs = rng.normal(cx, cluster_spread, size=c)
            ys = rng.normal(cy, cluster_spread, size=c)
            for x, y in zip(xs, ys):
                positions.append((float(np.clip(x, 0, 100)), float(np.clip(y, 0, 100))))

    return positions[:n]


def _build_star_gradients(
    n: int,
    seed: int,
    size_range: tuple[float, float] = (0.8, 2.6),
    opacity_range: tuple[float, float] = (0.45, 0.85),
    big_size_range: tuple[float, float] | None = (2.8, 4.8),
    big_prob: float = 0.18,
) -> str:
    """Return a comma-separated list of radial-gradient definitions for star dots."""
    rng = np.random.default_rng(seed)
    palette = [
        (247, 249, 253),  # ink white
        (245, 191, 60),   # orbital gold
        (106, 200, 255),  # accent cyan
    ]
    positions = _generate_star_positions(n, rng)

    gradients = []
    for i in range(n):
        if big_size_range and rng.random() < big_prob:
            size = rng.uniform(*big_size_range)
            alpha = rng.uniform(max(opacity_range[0], 0.65), 0.95)
        else:
            size = rng.uniform(*size_range)
            alpha = rng.uniform(*opacity_range)
        x_pct, y_pct = positions[i]
        r, g, b = palette[i % len(palette)]
        gradients.append(
            f"radial-gradient({size:.2f}px {size:.2f}px at {x_pct:.2f}% {y_pct:.2f}%, "
            f"rgba({r},{g},{b},{alpha:.2f}), transparent 55%)"
        )
    return ",\n            ".join(gradients)


STARFIELD_FAR = _build_star_gradients(
    280,
    seed=17,
    size_range=(0.75, 2.2),
    opacity_range=(0.50, 0.70),
    big_size_range=(2.4, 4.0),
    big_prob=0.14,
)
STARFIELD_NEAR = _build_star_gradients(
    180,
    seed=23,
    size_range=(1.1, 3.3),
    opacity_range=(0.65, 0.95),
    big_size_range=(3.2, 5.2),
    big_prob=0.22,
)

st.set_page_config(
    page_title="Prevent a Debris Cascade | Space Traffic Outlook",
    layout="wide",
    page_icon=":satellite:",
)

st.markdown(
    f"""
    <style>
    :root {{
        --ink: {APP_THEME["ink"]};
        --muted: {APP_THEME["muted"]};
        --surface: {APP_THEME["surface"]};
        --primary: {APP_THEME["primary"]};
        --accent: {APP_THEME["secondary"]};
    }}
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: var(--ink);
    }}
    .main .block-container {{
        padding-top: 0.1rem;
        max-width: 1400px;
        padding-bottom: 1.5rem;
        background: transparent;
    }}
    h1 {{ margin-bottom: 0.1rem; }}
    h2 {{ margin-top: 0.05rem; margin-bottom: 0rem; }}
    h3 {{ margin-top: 0.0rem; margin-bottom: 0.1rem; }}
    p, .stMarkdown {{ margin-bottom: 0.25rem; }}
    [data-testid="stCaption"] {{ margin-top: 0.05rem; margin-bottom: 0.15rem; }}
    hr {{ margin: 0.15rem 0 0.2rem 0; }}
    /* tighten select/slider spacing */
    .stSelectbox > div {{ padding-top: 0rem !important; padding-bottom: 0rem !important; margin-bottom: 0.1rem; }}
    .stSelectbox label {{ margin-bottom: 0rem; }}
    .stSlider > div {{ padding-top: 0rem !important; padding-bottom: 0rem !important; }}
    .stSlider label {{ margin-bottom: 0rem; }}
    .stColumn > div {{ padding-top: 0rem; padding-bottom: 0.05rem; }}
    div[data-testid="metric-container"] {{
        background: {APP_THEME["surface"]};
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.25);
    }}
    .metric-card {{
        background: #0f1626;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 9px 11px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        display: flex;
        flex-direction: column;
        gap: 6px;
        min-height: 110px;
        max-width: 360px;
        width: 100%;
        margin: 0 auto;
    }}
    .metric-card .title {{
        color: #cfd8e3;
        font-size: 0.9rem;
        font-weight: 600;
    }}
    .metric-card .value {{
        color: #f5bf3c;
        font-size: 1.28rem;
        font-weight: 700;
    }}
    .metric-card .delta {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #8ee0a3;
    }}
    .metric-dot {{
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
    }}
    .panel {{
        background: #0f1626;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        margin-bottom: 0.35rem;
    }}
    .panel.tight {{ padding: 10px 12px; }}
    body {{
        background: radial-gradient(circle at 10% 20%, rgba(245,191,60,0.06), transparent 28%),
                    radial-gradient(circle at 85% 10%, rgba(106,200,255,0.05), transparent 32%),
                    linear-gradient(180deg, #0b0f1a 0%, #0b1220 45%, #0c1324 100%);
        color: {APP_THEME["ink"]};
    }}
    [data-testid="stAppViewContainer"] {{
        position: relative;
        background: transparent;
        z-index: 1;
        isolation: isolate; /* keep star layers beneath content */
    }}
    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after {{
        content: "";
        position: fixed;
        inset: -160px; /* allow drift without revealing edges */
        pointer-events: none;
        background-repeat: no-repeat;
        background-size: 140% 140%;
        z-index: 0;
    }}
    [data-testid="stAppViewContainer"]::before {{
        background-image: {STARFIELD_FAR};
        opacity: 0.65;
        animation: starDriftFar 120s linear infinite, starPulse 8s ease-in-out infinite alternate;
    }}
    [data-testid="stAppViewContainer"]::after {{
        background-image: {STARFIELD_NEAR};
        opacity: 0.72;
        filter: drop-shadow(0 0 9px rgba(255,255,255,0.12));
        animation: starDriftNear 110s ease-in-out infinite, starPulse 6s ease-in-out infinite alternate;
    }}
    @keyframes starDriftFar {{
        from {{ transform: translate3d(0,0,0); }}
        to {{ transform: translate3d(-150px, -105px, 0); }}
    }}
    @keyframes starDriftNear {{
        0% {{ transform: translate3d(0,0,0); }}
        50% {{ transform: translate3d(150px, 105px, 0); }}
        100% {{ transform: translate3d(0,0,0); }}
    }}
    @keyframes starPulse {{
        from {{ opacity: 0.48; }}
        to {{ opacity: 1.00; }}
    }}
    h1, h2, h3, h4, h5 {{
        color: var(--ink);
        letter-spacing: -0.02em;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛰️ LEO Congestion Demands Debris Mitigation Now")
st.subheader("See how cleanup and explosion control reshape debris, collision risk, and cost.")
st.caption(
    "Data sources: SATCAT history, Nasa Orbital Debris Quarterly News, ESA Space Environment Report 9.1."
)
st.markdown("<hr style='margin-top:0.15rem; margin-bottom:0.25rem;'/>", unsafe_allow_html=True)
#guide_cols = st.columns(3)
#guide_cols[0].markdown("**How to read**\n\nBlue/teal = goal cleanup; amber/red = status quo. Thick line = median, band = uncertainty.")
#guide_cols[1].markdown("**Spotlight**\n\nYear slider locks every metric to the same point in time. Scenario pick drives all numbers.")
#guide_cols[2].markdown("**Money lens**\n\nLoss/spend sliders update dollar figures instantly—no extra math.")

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
        "label": "Goal cleanup | rare explosions",
        "tagline": "90% PMD; 0.10% explosion rate",
        "color": APP_THEME["secondary"],
        "pair_key": "Exp0.0010",
        "mitigated": True,
    },
    "PMD25_Exp0.0045": {
        "label": "Goal cleanup | current explosions",
        "tagline": "90% PMD; 0.45% explosion rate",
        "color": "#2fbf71",  # green to distinguish from other scenarios
        "pair_key": "Exp0.0045",
        "mitigated": True,
    },
    "NoMit_Exp0.0010": {
        "label": "Status quo | rare explosions",
        "tagline": "20% PMD; 0.10% explosion rate",
        "color": APP_THEME["amber"],
        "pair_key": "Exp0.0010",
        "mitigated": False,
    },
    "NoMit_Exp0.0045": {
        "label": "Status quo | current explosions",
        "tagline": "20% PMD; 0.45% explosion rate",
        "color": APP_THEME["scarlet"],
        "pair_key": "Exp0.0045",
        "mitigated": False,
    },
}

PLOT_FONT = dict(family="Inter, 'Segoe UI', sans-serif", size=14, color="#e5edff")
PLOT_BG = "#0c1324"
GRID_COLOR = "#1f2937"


def sentiment_from_delta(
    delta: float | None,
    better_when: str = "lower",
    neutral_band: float = 0.05,
    anchor_best: bool = False,
    near_worst_band: float | None = None,
) -> str:
    """
    Return sentiment bucket for coloring: good/neutral/bad.

    anchor_best=True treats parity with the reference (delta ~ 0) as good.
    neutral_band is the tolerance band (fractional, e.g., 0.05 = 5%).
    near_worst_band marks a red zone near the worst case for higher-is-better metrics.
    """
    if delta is None:
        return "neutral"

    if anchor_best:
        if better_when == "higher":
            if delta >= 0:
                return "good"
            if delta >= -neutral_band:
                return "neutral"
            return "bad"
        # lower is better
        if delta <= 0:
            return "good"
        if delta <= neutral_band:
            return "neutral"
        return "bad"

    if better_when == "higher":
        if near_worst_band is not None and delta <= near_worst_band:
            return "bad"
        if delta >= neutral_band:
            return "good"
        if delta <= -neutral_band:
            return "bad"
        return "neutral"

    # lower is better
    if delta <= -neutral_band:
        return "good"
    if delta >= neutral_band:
        return "bad"
    return "neutral"


def render_metric_card(title: str, value: str, delta_text: str | None, sentiment: str):
    """Render a custom metric card with sentiment-colored delta."""
    color = SENTIMENT_COLORS.get(sentiment, SENTIMENT_COLORS["neutral"])
    delta_txt = delta_text or "No change"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="title">{title}</div>
            <div class="value">{value}</div>
            <div class="delta" style="color:{color}">
                <span class="metric-dot" style="background:{color}"></span>{delta_txt}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------
# USER INPUTS (SIDEBAR)
# ----------------------------------
st.sidebar.header("Scenario inputs")
st.sidebar.write(
    "Set cleanup compliance (PMD) and explosion likelihood to compare futures."
)

# Hard-coded defaults for path and simulation sizing (previous sidebar controls)
DATA_PATH = "./Data"
satcat_filename = "satcat.csv"
HIST_TAIL = 25
N_YEARS_FWD = 200
N_PATHS = 300

selected_scenarios = st.sidebar.multiselect(
    "Choose scenarios",
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
    help="Pick at least one. The color legend matches the charts.",
)

with st.sidebar.expander("Scenario definitions", expanded=False):
    for key in SCENARIOS:
        meta = SCENARIO_META[key]
        st.markdown(f"**{meta['label']}** - {meta['tagline']}")

st.sidebar.header("Money lens")
COLLISION_COST_MUSD = st.sidebar.slider(
    "Loss per collision event (USD millions)",
    min_value=50,
    max_value=1000,
    value=250,
    step=25,
    help="Select value each catastrophic collision.",
)
AVOIDANCE_COST_MUSD = st.sidebar.slider(
    "Spend per collision avoidance (USD millions)",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    help="Budgeted cost to avoid one collision (e.g., deorbit or maneuver spend).",
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



def make_orbital_rings_plotly(df: pd.DataFrame):
    """Interactive sunburst fallback with clearer styling."""
    agg = df.groupby(["Band", "Group"], as_index=False)["Count"].sum()
    agg["Count"] = pd.to_numeric(agg["Count"], errors="coerce").fillna(0)
    fig = px.sunburst(
        agg,
        path=["Band", "Group"],
        values="Count",
        color="Group",
        color_discrete_map={"Debris": "#e94b3c", "Satellites": "#3da5ff"},
        height=760,
        branchvalues="total",
    )
    fig.update_traces(
        insidetextorientation="radial",
        textinfo="label+percent parent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,.0f}<br>% of parent: %{percentParent:.1%}<extra></extra>",
        marker=dict(line=dict(color="#ffffff", width=1.2), colorscale=None),
    )
    fig.update_traces(root_color=APP_THEME["amber"], selector=dict(type="sunburst"))
    if fig.data:
        trace = fig.data[0]
        colors = list(getattr(trace.marker, "colors", []))
        parents = list(getattr(trace, "parents", []))
        if colors and parents:
            for idx, parent in enumerate(parents):
                if parent == "":
                    colors[idx] = APP_THEME["amber"]
            trace.marker.colors = colors
    fig.update_layout(
        margin=dict(t=60, l=0, r=0, b=0),
        title="Distribution of Satellites and Debris: Risk of Cluttered Bands",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor="#ffffff",
        font=PLOT_FONT,
        uniformtext_minsize=12,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        sunburstcolorway=[APP_THEME["amber"], "#3da5ff", "#e94b3c"],
        extendsunburstcolors=True,
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

def add_fan_traces(fig, years, paths, meta, show_band=True, y_fmt=",.0f"):
    """Add median + 5-95% band for a scenario with consistent hover styling."""
    q = quantiles_over_time(paths)
    legendgroup = meta["label"]
    if show_band:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=q["p95"],
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name=None,
                legendgroup=legendgroup,
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
                legendgroup=legendgroup,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=q["p50"],
            mode="lines",
            line=dict(color=meta["color"], width=4),
            name=meta["label"],
            legendgroup=legendgroup,
            hovertemplate=f"{meta['label']}<br>Year %{{x}}<br>%{{y:{y_fmt}}}<extra></extra>",
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

    st.markdown("### Scenario and Year Spotlight ✨", help=None)
    focus_cols = st.columns([2, 1], gap="small")
    with focus_cols[0]:
        focus_scenario = st.selectbox(
            label="Scenario spotlight",
            options=selected_scenarios,
            format_func=lambda x: SCENARIO_META[x]["label"],
            help="Use colors to link the selection with the charts.",
            key="focus_scenario",
            label_visibility="collapsed",
        )
    with focus_cols[1]:
        focus_year = st.slider(
            label="Year in focus",
            min_value=int(years[0]),
            max_value=int(years[-1]),
            value=default_focus_year,
            step=1,
            help="Scroll to any year in the projection window.",
            label_visibility="collapsed",
        )

    focus_pack = fans[focus_scenario]
    obj_summary = summarize_paths_at_year(years, focus_pack["N"], focus_year)
    coll_summary = summarize_paths_at_year(years, focus_pack["C"], focus_year)

    best_obj_summary = summarize_paths_at_year(best_pack["YEARS"], best_pack["N"], focus_year)
    best_coll_summary = summarize_paths_at_year(best_pack["YEARS"], best_pack["C"], focus_year)

    worst_obj_summary = summarize_paths_at_year(worst_pack["YEARS"], worst_pack["N"], focus_year)
    worst_coll_summary = summarize_paths_at_year(worst_pack["YEARS"], worst_pack["C"], focus_year)

    st.markdown(f"#### Decide for {obj_summary['year']}: Act on these Conditions 🎯")
    kpi_cols = st.columns(3, gap="small")

    obj_delta_vs_best = pct_delta(obj_summary["median"], best_obj_summary["median"])
    obj_sent = sentiment_from_delta(obj_delta_vs_best, better_when="lower", neutral_band=0.25, anchor_best=True)
    with kpi_cols[0]:
        render_metric_card(
            f"Objects crowding orbit in {obj_summary['year']}",
            f"{obj_summary['median']:,.0f}",
            (
                f"{format_pct(obj_delta_vs_best, decimals=1, show_sign=True)} vs best mitigation"
                if obj_delta_vs_best is not None else None
            ),
            obj_sent,
        )

    coll_delta_vs_best = pct_delta(coll_summary["median"], best_coll_summary["median"])
    coll_sent = sentiment_from_delta(coll_delta_vs_best, better_when="lower", neutral_band=0.50, anchor_best=True)
    with kpi_cols[1]:
        render_metric_card(
            f"Catastrophic collisions on this path by {coll_summary['year']}",
            f"{coll_summary['median']:.2f}",
            (
                f"{format_pct(coll_delta_vs_best, decimals=1, show_sign=True)} vs best mitigation"
                if coll_delta_vs_best is not None else None
            ),
            coll_sent,
        )

    collisions_reduced_vs_worst = worst_coll_summary["median"] - coll_summary["median"]
    best_collisions_reduced_vs_worst = worst_coll_summary["median"] - best_coll_summary["median"]
    avoid_delta_vs_best = (
        pct_delta(collisions_reduced_vs_worst, best_collisions_reduced_vs_worst)
        if best_collisions_reduced_vs_worst > 0 else None
    )
    avoid_sent_top = sentiment_from_delta(
        avoid_delta_vs_best,
        better_when="higher",
        neutral_band=0.30,
        near_worst_band=0.05,
        anchor_best=True,
    )
    with kpi_cols[2]:
        render_metric_card(
            "Collisions you avoid vs worst-case",
            f"{collisions_reduced_vs_worst:.2f}",
            (
                f"{format_pct(avoid_delta_vs_best, decimals=1, show_sign=True)} vs best mitigation"
                if avoid_delta_vs_best is not None else None
            ),
            avoid_sent_top,
        )

    st.markdown("#### Justify Mitigation with Money and Risk 💰")
    fin_cols = st.columns(2, gap="small")
    expected_loss = coll_summary["median"] * COLLISION_COST_MUSD
    best_expected_loss = best_coll_summary["median"] * COLLISION_COST_MUSD
    exp_loss_delta = pct_delta(expected_loss, best_expected_loss) if best_expected_loss > 0 else None
    exp_sent = sentiment_from_delta(exp_loss_delta, better_when="lower", neutral_band=0.50, anchor_best=True)
    with fin_cols[0]:
        render_metric_card(
            "Median loss you must budget for",
            format_musd(expected_loss),
            (
                f"{format_pct(exp_loss_delta, decimals=1, show_sign=True)} vs best mitigation"
                if exp_loss_delta is not None else None
            ),
            exp_sent,
        )

    reduction_pct_vs_worst = None
    if worst_coll_summary["median"] > 0:
        reduction_pct_vs_worst = collisions_reduced_vs_worst / worst_coll_summary["median"]
    collisions_avoided = max(0.0, collisions_reduced_vs_worst)
    loss_avoided_vs_worst = collisions_avoided * COLLISION_COST_MUSD
    best_loss_avoided_vs_worst = max(0.0, worst_coll_summary["median"] - best_coll_summary["median"]) * COLLISION_COST_MUSD
    loss_avoid_delta_vs_best = (
        pct_delta(loss_avoided_vs_worst, best_loss_avoided_vs_worst)
        if best_loss_avoided_vs_worst > 0 else None
    )
    avoidance_budget = collisions_avoided * AVOIDANCE_COST_MUSD
    net_benefit = loss_avoided_vs_worst - avoidance_budget
    roi_pct = pct_delta(loss_avoided_vs_worst, avoidance_budget) if avoidance_budget > 0 else None
    roi_sent = "good" if net_benefit > 0 else ("neutral" if net_benefit == 0 else "bad")
    loss_sent = sentiment_from_delta(
        loss_avoid_delta_vs_best,
        better_when="higher",
        neutral_band=0.30,
        near_worst_band=0.10,
        anchor_best=True,
    )
    combined_sentiment = "bad" if roi_sent == "bad" else loss_sent
    with fin_cols[1]:
        render_metric_card(
            "Mitigation value unlocked vs worst-case",
            format_musd(loss_avoided_vs_worst),
            (
                f"Net after spend: {format_musd(net_benefit)} | ROI: {format_pct(roi_pct, decimals=1, show_sign=True)}"
                if roi_pct is not None else f"Net after spend: {format_musd(net_benefit)}"
            ),
            combined_sentiment,
        )

    st.markdown("### Choose Mitigation to Bend the Projected Futures 📈")

    st.caption("Shaded bands show the range of possible Monte Carlo runs; solid lines are medians.")
    obj_fig = go.Figure()
    for name, pack in fans.items():
        add_fan_traces(obj_fig, pack["YEARS"], pack["N"], SCENARIO_META[name], y_fmt=",.0f")
    obj_fig.update_layout(
        title=dict(text="Mitigation Bends the Large-Object Population Curve (<10 cm)", y=0.99, yanchor="top", pad=dict(t=6)),
        xaxis_title="Year",
        yaxis_title="Objects in orbit (effective count)",
        hovermode="x unified",
        template="none",
        legend_title="Scenarios",
        height=700,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor="#0f172a",
        font=PLOT_FONT,
        margin=dict(t=80, l=85, r=20, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    obj_fig.update_yaxes(tickformat=",", gridcolor=GRID_COLOR, zeroline=False, title_standoff=28, color="#e5edff")
    obj_fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, dtick=25, title_standoff=16, color="#e5edff")
    st.plotly_chart(obj_fig, width="stretch", config={"displayModeBar": False})

    coll_fig = go.Figure()
    for name, pack in fans.items():
        add_fan_traces(coll_fig, pack["YEARS"], pack["C"], SCENARIO_META[name], y_fmt=",.2f")
    coll_fig.update_layout(
        title=dict(text="Active Cleanup Slows Catastrophic Collisions (cumulative)", y=0.99, yanchor="top", pad=dict(t=6)),
        xaxis_title="Year",
        yaxis_title="Cumulative catastrophic collisions",
        hovermode="x unified",
        template="none",
        legend_title="Scenarios",
        height=700,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor="#0f172a",
        font=PLOT_FONT,
        margin=dict(t=80, l=85, r=20, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    coll_fig.update_yaxes(tickformat=",.2f", gridcolor=GRID_COLOR, zeroline=False, title_standoff=28, color="#e5edff")
    coll_fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, dtick=25, title_standoff=16, color="#e5edff")
    st.plotly_chart(coll_fig, width="stretch", config={"displayModeBar": False})

    st.markdown("### Defuse Cascade Risk: Protect Orbital Assets and Drain Debris 🧹")
    orbital_df = load_orbital_shell_counts(DATA_PATH)
    if orbital_df is None or orbital_df.empty:
        st.info("Orbital shell counts not found. Add `orbital_shell_counts_long.csv` to the Data folder to view the orbital congestion rings.")
    else:
        orbital_df = prep_orbital_bands(orbital_df)
        band_totals = orbital_df.groupby("Band")["Count"].sum().sort_values(ascending=False)
        default_band = band_totals.index[0] if not band_totals.empty else None

        fig_orb = make_orbital_rings_plotly(orbital_df) 
        st.plotly_chart(fig_orb, width="stretch", config={"displayModeBar": False})

        # KPI cards for selected band
        band_choice = st.selectbox(
            "Pick the next band to protect/clean",
            options=list(band_totals.index),
            index=0 if default_band else None,
            help="Bands are altitude slices. Start with the most crowded band.",
        )

        band_slice = orbital_df[orbital_df["Band"] == band_choice]
        band_total = float(band_slice["Count"].sum())
        debris_count = float(band_slice.loc[band_slice["Group"] == "Debris", "Count"].sum())
        sat_count = float(band_total - debris_count)
        debris_share = (debris_count / band_total) if band_total > 0 else 0.0
        total_all = float(orbital_df["Count"].sum())
        band_global_share = (band_total / total_all) if total_all > 0 else 0.0
        crowd_rank = list(band_totals.index).index(band_choice) + 1 if band_choice in band_totals.index else None

        st.markdown("#### Band Snapshot: Safeguard our Infrastructure 📸")
        bcols = st.columns(3)
        with bcols[0]:
            render_metric_card(
                "Infrastructure share at stake",
                f"{band_global_share*100:.1f}%",
                f"Objects in band: {band_total:,.0f}",
                "neutral",
            )
        with bcols[1]:
            render_metric_card(
                "Debris share",
                f"{debris_share*100:.1f}%",
                f"Debris: {debris_count:,.0f} | Satellites: {sat_count:,.0f}",
                "neutral",
            )
        with bcols[2]:
            rank_text = f"#{crowd_rank} most crowded" if crowd_rank is not None else "Unranked"
            render_metric_card(
                "Crowding rank",
                rank_text,
                f"Most crowded: {band_totals.index[0]}",
                "neutral",
            )

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
        color_discrete_map={"Debris": "#e94b3c", "Satellites": "#3da5ff"},
        barmode="stack",
        title="Debris vs Satellites: Focus Band Compared to the Most and Least Crowded",
        height=360,
    )
    fig_mini.update_layout(
        legend_title="Object type",
        xaxis_title=None,
        yaxis_title="Objects",
        font=PLOT_FONT,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor="#0f172a",
        margin=dict(t=60, l=40, r=10, b=40),
    )
    st.plotly_chart(fig_mini, width="stretch", config={"displayModeBar": False})
