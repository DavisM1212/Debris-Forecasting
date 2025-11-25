import pandas as pd
import numpy as np

# File definitions
DATA_PATH = './Data/'
INPUT = 'satcat.csv'
OUT_WIDE = 'orbital_shell_counts_wide.csv'
OUT_LONG = 'orbital_shell_counts_long.csv'
OUT_SUMMARY = 'orbital_shell_summary.csv'

# Contiguous orbital bands for SMA >= 0 (better not be 0...)
BANDS = [
    ('LEO 0-400 km',             0,       400),
    ('LEO 400-700 km',           400,     700),
    ('LEO 700-1200 km',          700,     1200),
    ('MEO 1200-19000 km',        1200,    19000),
    ('GNSS 19000-23000 km',      19000,   23000),
    ('High incl GEO 23000+ km',  23000,   np.inf),
]

# Map to readable labels
TYPE_MAP = {'PAY': 'Satellites', 'DEB': 'Debris', 'R/B': 'Rocket_Bodies'}

# Orbital constants
MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM      = 6378.137
SEC_PER_DAY     = 86400.0
TWO_PI          = 2.0 * np.pi
MIN_PER_DAY     = 1440.0

# Load and filter needed columns
df = pd.read_csv(DATA_PATH + INPUT)

df["OBJECT_TYPE"] = df["OBJECT_TYPE"].astype(str).str.strip().str.upper()
keep_types = set(TYPE_MAP.keys())
df = df[df["OBJECT_TYPE"].isin(keep_types)]

if "OPS_STATUS_CODE" in df.columns:
    df = df[df["OPS_STATUS_CODE"].astype(str).str.strip().str.upper() != "D"]
if "ORBIT_CENTER" in df.columns:
    df = df[df["ORBIT_CENTER"].astype(str).str.strip().str.upper() == "EA"]
if "ORBIT_TYPE" in df.columns:
    df = df[df["ORBIT_TYPE"].astype(str).str.strip().str.upper() == "ORB"]

# Coerce numeric data types
for c in ["APOGEE", "PERIGEE", "PERIOD"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Compute SMA altitude, prefer using apogee and perigee but will use period
# if there are valid records that don't ahve apogee and perigee data
use_ap_per = df["APOGEE"].notna() & df["PERIGEE"].notna()
if "APOGEE" in df.columns and "PERIGEE" in df.columns:
    sma_gee = (df["APOGEE"] + df["PERIGEE"]) / 2.0
else:
    sma_gee = pd.Series(np.nan, index=df.index)

if "PERIOD" in df.columns:
    val_per = df["PERIOD"].notna() & (df["PERIOD"] > 0)
    n_rev_day = pd.Series(np.nan, index=df.index)
    n_rev_day[val_per] = MIN_PER_DAY / df.loc[val_per, "PERIOD"]
    n_rad_s = n_rev_day * TWO_PI / SEC_PER_DAY
    a_km = (MU_EARTH_KM3_S2 / (n_rad_s ** 2)) ** (1.0 / 3.0)
    sma_per = a_km - R_EARTH_KM
else:
    sma_per = pd.Series(np.nan, index=df.index)

df["SMA"] = np.where(use_ap_per, sma_gee, sma_per)
df = df.dropna(subset=["SMA"])
df = df[(df["SMA"] >= 0) & (df["SMA"] <= 2.0e5)]

# Map object types to readable for aggregation
df["Object_Type"] = df["OBJECT_TYPE"].map(TYPE_MAP)

# Bin into orbital bands
edges = [lo for (_, lo, _) in BANDS] + [BANDS[-1][2]]
labels = [lbl for (lbl, _, _) in BANDS]

df['Band'] = pd.cut(
    df['SMA'],
    bins=edges,
    labels=labels,
    right=False,          # [low, high)
    include_lowest=True
)

# Drop any rows that somehow fell outside (shouldn't happen for SMA>=0)
df = df.dropna(subset=['Band'])

# Aggregate for saving
# Wide
wide = (
    df.pivot_table(index='Band', columns='Object_Type', values='SMA',
                   aggfunc='count', fill_value=0)
      .astype(int)
      .reindex(labels)  # keep configured order
)

# Ensure consistent columns for charts
for col in ['Satellites', 'Debris', 'Rocket_Bodies']:
    if col not in wide.columns:
        wide[col] = 0

wide['Total'] = wide[['Satellites', 'Debris', 'Rocket_Bodies']].sum(axis=1)

# Long
long = (
    wide.reset_index()
        .melt(id_vars='Band',
              value_vars=['Satellites', 'Debris', 'Rocket_Bodies'],
              var_name='Object_Type', value_name='Count')
)

# Summary
summary = wide.reset_index().copy()
summary['Debris_Share'] = (summary['Debris'] / summary['Total']).replace([np.inf, np.nan], 0.0)
summary['Debris_Share_Pct'] = (summary['Debris_Share'] * 100).round(1)


wide.to_csv(DATA_PATH + OUT_WIDE, index=True, lineterminator='\n')
long.to_csv(DATA_PATH + OUT_LONG, index=False, lineterminator='\n')
summary.to_csv(DATA_PATH + OUT_SUMMARY, index=False, lineterminator='\n')

print('Wrote:')
print(' -', DATA_PATH + OUT_WIDE)
print(' -', DATA_PATH + OUT_LONG)
print(' -', DATA_PATH + OUT_SUMMARY)