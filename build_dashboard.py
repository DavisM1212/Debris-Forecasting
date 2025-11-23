import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_helpers import (
    growth_rate_annualized,
    pct_delta,
    quantiles_over_time,
    summarize_paths_at_year,
)

# ----------------------------------
# STREAMLIT CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Orbital Debris Projection",
    layout="wide"
)

st.title("Orbital Debris Monte Carlo Projections (<10 cm)")
st.write("""
Interactive Monte Carlo forecasts of large orbital debris and catastrophic collisions
under different mitigation scenarios, based on SATCAT history.
""")

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
    "Scenarios to plot",
    options=[
        "NoMit_Exp0.0045",
        "PMD25_Exp0.0045",
        "NoMit_Exp0.0010",
        "PMD25_Exp0.0010",
    ],
    default=[
        "NoMit_Exp0.0045",
        "PMD25_Exp0.0045",
        "NoMit_Exp0.0010",
        "PMD25_Exp0.0010",
    ]
)

st.sidebar.header("Financial Assumptions")
COLLISION_COST_MUSD = st.sidebar.slider(
    "Cost per catastrophic collision (USD millions)",
    min_value=50, max_value=1000, value=250, step=25
)
AVOIDANCE_COST_MUSD = st.sidebar.slider(
    "Avoidance/mitigation spend per collision avoided (USD millions)",
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

SCENARIOS = {
    "NoMit_Exp0.0045": {"pmd_comp": 0.20, "P8": 0.0045},
    "PMD25_Exp0.0045": {"pmd_comp": 0.90, "P8": 0.0045},
    "NoMit_Exp0.0010": {"pmd_comp": 0.20, "P8": 0.0010},
    "PMD25_Exp0.0010": {"pmd_comp": 0.90, "P8": 0.0010},
}

SCENARIO_META = {
    "NoMit_Exp0.0045": {
        "label": "No Mitigation - 0.45% Expl Rate",
        "color": "#A50021",
        "pair_key": "Exp0.0045",
        "mitigated": False,
    },
    "PMD25_Exp0.0045": {
        "label": "PMD - 0.45% Expl Rate",
        "color": "#2E8B57",
        "pair_key": "Exp0.0045",
        "mitigated": True,
    },
    "NoMit_Exp0.0010": {
        "label": "No Mitigation - 0.10% Expl Rate",
        "color": "#FF7043",
        "pair_key": "Exp0.0010",
        "mitigated": False,
    },
    "PMD25_Exp0.0010": {
        "label": "PMD - 0.10% Expl Rate",
        "color": "#4682B4",
        "pair_key": "Exp0.0010",
        "mitigated": True,
    },
}


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
    A = np.trapz(base(t), t)
    A = max(A, 1e-12)
    return -np.log(1.0 - P8_target)/A

def yearly_hazard_matrix(class_name, k_scale, max_age=200):
    base = LAM_BASE[class_name]
    ages = np.arange(0, max_age+1)
    grid = np.linspace(0, DAY, 721)
    H = np.zeros_like(ages, dtype=float)
    for a in ages:
        tt = a*DAY + grid
        H[a] = k_scale * np.trapz(base(tt), tt)
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

st.markdown(
    f"**Baseline year:** {BASELINE_YEAR}  "
    f"(SC alive: {int(cohorts0['SC'].sum()):,}, "
    f"RB alive: {int(cohorts0['RB'].sum()):,}, "
    f"SOZ alive: {int(cohorts0['SOZ'].sum()):,})"
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
    TARGET_BASELINE_EFF = 1.2e4
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

    years = next(iter(fans.values()))["YEARS"]
    default_focus_year = int(min(years[0] + 50, years[-1]))

    focus_scenario = st.selectbox(
        "Scenario for KPI focus",
        selected_scenarios,
        format_func=lambda x: SCENARIO_META[x]["label"],
    )
    focus_year = st.slider(
        "Insight year",
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

    pair_name = find_pair_scenario(focus_scenario)
    collisions_avoided = None
    cost_avoided_musd = None
    objects_delta = None
    coll_pct_reduction = None
    if pair_name and pair_name in fans:
        pair_pack = fans[pair_name]
        pair_obj = summarize_paths_at_year(years, pair_pack["N"], focus_year)
        pair_coll = summarize_paths_at_year(years, pair_pack["C"], focus_year)

        if SCENARIO_META[focus_scenario]["mitigated"]:
            mitig_obj, mitig_coll = obj_summary, coll_summary
            base_obj, base_coll = pair_obj, pair_coll
        else:
            mitig_obj, mitig_coll = pair_obj, pair_coll
            base_obj, base_coll = obj_summary, coll_summary

        collisions_avoided = base_coll["median"] - mitig_coll["median"]
        objects_delta = base_obj["median"] - mitig_obj["median"]
        cost_avoided_musd = max(0.0, collisions_avoided) * COLLISION_COST_MUSD
        coll_pct_reduction = pct_delta(mitig_coll["median"], base_coll["median"])

    kpi_cols = st.columns(3)
    kpi_cols[0].metric(
        f"{SCENARIO_META[focus_scenario]['label']} objects @ {obj_summary['year']}",
        f"{obj_summary['median']:,.0f}",
        delta=f"P95: {obj_summary['p95']:,.0f}"
    )
    kpi_cols[1].metric(
        f"Cumulative collisions @ {coll_summary['year']}",
        f"{coll_summary['median']:.2f}",
        delta=f"P95: {coll_summary['p95']:.2f}"
    )
    kpi_cols[2].metric(
        "Annualized growth rate",
        f"{growth*100:.2f}%",
        delta=f"Vs baseline: {obj_summary['median']-base_obj_median:,.0f} objects"
    )

    fin_cols = st.columns(3)
    fin_cols[0].metric(
        "Collision exposure (median)",
        format_cost_musd(coll_summary["median"] * COLLISION_COST_MUSD),
        delta=f"P95: {format_cost_musd(coll_summary['p95'] * COLLISION_COST_MUSD)}"
    )

    if collisions_avoided is not None:
        delta_parts = []
        if objects_delta is not None:
            delta_parts.append(f"{objects_delta:,.0f} fewer objects")
        if coll_pct_reduction is not None:
            delta_parts.append(f"{coll_pct_reduction*100:+.1f}% vs paired")
        delta_text = " | ".join(delta_parts) if delta_parts else None

        fin_cols[1].metric(
            "Collisions avoided vs paired scenario",
            f"{collisions_avoided:.2f}",
            delta=delta_text
        )
        fin_cols[2].metric(
            "Cost avoided",
            format_cost_musd(cost_avoided_musd),
            delta=f"Spend (@{format_cost_musd(AVOIDANCE_COST_MUSD)} ea): {format_cost_musd(max(0.0, collisions_avoided) * AVOIDANCE_COST_MUSD)}"
        )
    else:
        fin_cols[1].metric("Collisions avoided vs paired scenario", "N/A", delta="Pair not selected")
        fin_cols[2].metric("Cost avoided", "N/A", delta=None)

    st.markdown("---")

    obj_fig = go.Figure()
    for name, pack in fans.items():
        add_fan_traces(obj_fig, pack["YEARS"], pack["N"], SCENARIO_META[name])
    obj_fig.update_layout(
        title="Projected Effective Number of Objects (<10 cm)",
        xaxis_title="Year",
        yaxis_title="Objects (<10 cm)",
        hovermode="x unified",
        template="simple_white",
        legend_title="Scenarios",
    )
    st.plotly_chart(obj_fig, use_container_width=True, config={"displayModeBar": False})

    coll_fig = go.Figure()
    for name, pack in fans.items():
        add_fan_traces(coll_fig, pack["YEARS"], pack["C"], SCENARIO_META[name])
    coll_fig.update_layout(
        title="Projected Cumulative Catastrophic Collisions",
        xaxis_title="Year",
        yaxis_title="Cumulative catastrophic collisions",
        hovermode="x unified",
        template="simple_white",
        legend_title="Scenarios",
    )
    st.plotly_chart(coll_fig, use_container_width=True, config={"displayModeBar": False})
