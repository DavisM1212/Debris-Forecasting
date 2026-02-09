# Debris-Forecasting Dashboard

> A Streamlit-powered decision dashboard that quantifies space-debris risk, compares mitigation policies, and pinpoints which orbital bands to protect first.

**Live Version:** (Debris Forecasting Dashboard)[https://satellite-debris-forecasting.streamlit.app/]
(Streamlit puts a time limit on the dashboard. If it is down contact me and I will bring it back up.)

---

## Why This Matters

- **Cascade risk is real:** Crowded low Earth orbit (LEO) bands can trigger self-sustaining collision chains that deny access to space for decades.  
- **High-value assets at stake:** Human spaceflight, science payloads, Earth observation, and ~$300B in GEO infrastructure all depend on predictable, low-risk orbits.  
- **Policy levers exist today:** Stricter post-mission disposal, explosion prevention, and targeted debris removal materially bend the future collision curve. This dashboard shows how much and where to act.

## What the Dashboard Shows

- **Scenario & Year Spotlight:** Choose mitigation vs. status-quo scenarios and synchronize every chart to a single year to see the consequences in one view.  
- **Decision KPIs:** Objects in orbit, cumulative catastrophic collisions, and avoided collisions - plus the associated dollar impacts (expected loss vs. mitigation value).  
- **Projected Futures:** Monte Carlo fans for large-object populations and cumulative catastrophic collisions, highlighting uncertainty and the payoff of mitigation.  
- **Orbital Band Focus:** Sunburst and band-level KPIs reveal where debris dominates, where most infrastructure sits (400–700 km), and where upper-band debris (700–1200 km) decays into crowded LEO. A comparative bar shows your focus band against the most and least crowded slices.  
- **Mitigation Narrative:** Titles and labels are action-oriented to keep attention on decisions: protect 400–700 km assets, drain 700–1200 km debris sources, and monitor permanent GNSS/GEO orbits.

## Data & Methodology

- **Sources:** Historic object counts from SATCAT; debris trends from NASA Orbital Debris Quarterly News; environment context from ESA Space Environment Report 9.1.  
- **Simulation:** Monte Carlo fans over 200-year windows, varying explosion probabilities, post‑mission disposal compliance, fragmentation decay, and solar cycle effects. Best vs. worst cases bracket realistic policy envelopes.  
- **Band Modeling:** Orbital shell counts grouped into altitude bands; debris vs. satellite shares computed and visualized (sunburst + stacked bars) to surface where cleanup or launch discipline matters most.  
- **Economic Lens:** Collision cost and avoidance cost (USD millions) propagate through KPIs to show expected loss, ROI of mitigation, and net benefit after spend.  
- **Storytelling Choices:** Preattentive color cues map to mitigation vs. status quo; concise, directive titles tie visuals to actions.

## Getting Started

### Prerequisites
- Python 3.12+ (repo tested with the provided `.venv`)
- Node not required; everything runs via Streamlit + Plotly.

### Install & Run
```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
streamlit run build_dashboard.py
```
The dashboard opens at `http://localhost:8501`. Use the sidebar to select scenarios and adjust the focus year.

### Tests
```bash
.venv\Scripts\activate
python -m pytest
```
All helper utilities are covered; run tests after changes to keep calculations trustworthy.

## Using the Dashboard (Demo Flow)

1. **Pick scenarios:** Start with best-mitigation vs. status-quo to maximize contrast.  
2. **Set year:** Slide to 2075; all KPIs and charts sync.  
3. **Read KPIs:** Note orbit objects, catastrophic collisions, and avoided collisions; check dollar impacts.  
4. **Scan projections:** Monte Carlo fans - look for flattening with mitigation.  
5. **Target bands:** Select 400–700 km (infrastructure-heavy) and 700–1200 km (debris source). Review band KPIs and the focus-vs-extremes bar.  
6. **Close with action:** Emphasize that policy/enforcement is the lever - mitigation measurably reduces collisions and protects assets.

## Feature Glossary

- **Scenario Controls:** Four baseline scenarios combining post‑mission disposal compliance and explosion rates; easily extendable.  
- **Decision KPIs:** Dynamic cards for orbit objects, collisions, avoided collisions, expected loss, mitigation value, and ROI.  
- **Projected Futures Charts:** Plotly fans with unified hover; uncertainty bands + median lines.  
- **Orbital Band Explorer:** Sunburst for debris/satellite mix; band selector; triage KPIs; stacked bar comparing focus vs. most/least crowded bands.  
- **Styling:** Custom theme, starfield background, preattentive palette; action-oriented titles to keep stakeholders on the “so what.”

## Project Structure

- `build_dashboard.py` – Streamlit app, layouts, figures, KPI logic.  
- `dashboard_helpers.py` – Formatting helpers, quantile computations, shared utilities.  
- `create_orbitals.py`, `build_shell_counts.py` – Data prep for legacy orbital shell count plotting. Not strictly needed.  
- `Data/` – Expected location for `orbital_shell_counts_long.csv` and other inputs.  
- `requirements.txt` – Python dependencies.

## Assumptions & Limits

- Focuses on tracked objects down to ~10 cm; smaller debris not modeled and numbers in the millions. 
- Economic figures use configurable cost assumptions; adjust to align with current valuations.  
- Band definitions are altitude-based; GNSS/GEO permanence highlighted but not decay-modeled (they’re effectively permanent).

## Extending the Work

- Plug in updated SATCAT/ESA/NASA releases for fresher baselines.  
- Add active-removal levers (objects/year) to test remediation pacing.  
- Integrate launch manifests to stress-test proposed constellation growth.  
- Export comparison reports (PDF/PNG) for policy briefings.

---
**Quick takeaway:** Mitigation isn’t speculative - the dashboard quantifies how targeted cleanup and explosion control flatten collision risk, protect critical LEO infrastructure, and safeguard permanent GEO/GNSS assets. The next move is enforcement. 
