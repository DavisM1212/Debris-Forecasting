import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
import math

# File path definitions
DATA_PATH = './Data/'
INPUT      = 'orbital_shell_counts_long.csv'
OUT_PNG    = 'orbital_rings_6band_curvedlabels.png'
OUT_PNG_T  = 'orbital_rings_6band_curvedlabels_transparent.png'

# Layout and scaling constants for the rings
BASE_RADIUS = 1.0
RING_SPACING = 0.70
THICKNESS_MIN = 0.25
THICKNESS_MAX_EXTRA = 1.60
SCALING_MODE = 'power'       # Allow scaling modes: 'power' | 'linear' | 'log'
POWER_EXP = 1.3
START_ANGLE_DEG = 90.0
ALPHA = 0.92
FIGSIZE = (10, 10)
MARGIN = 0.60

# Fonts and text controls grouped together
TITLE_FONT = dict(fontsize=18, fontweight='bold', family='Arial', color='white')
LABEL_FP   = FontProperties(family='Arial', weight='bold', size=12)
LEGEND_FONT_SIZE = 12
LEGEND_TITLE_SIZE = 13
COLOR_MAP = {'Debris': '#d62728', 'Satellites': '#1f77b4'}
TEXT_OUTLINE = [pe.withStroke(linewidth=3, foreground='white')]
BG_COL = '#1e1f24'

# Curved label spacing boost between characters as a fraction of average spacing
CHAR_SPACING_BOOST = 0.10

# Arc sizing bounds for curved labels
CHAR_DEG_AT_R1 = 13.0     # Extra boost for the first label when it gets squished
MIN_ARC_DEG    = 40       # Minimum to keep short labels readable
MAX_ARC_DEG    = 120      # Cap to prevent wild wrapping

# Compute an arc that fits the text at radius r
def compute_arc_deg(text: str, r: float) -> float:
    n = max(1, len(text))
    # Scale per-character degrees roughly with 1/r
    deg_per_char = CHAR_DEG_AT_R1 / max(0.2, r)  # Guard against very small r
    # Treat the label as (n-1) gaps and add one more for padding
    raw = (n + 0) * deg_per_char * (1.0 + CHAR_SPACING_BOOST)
    return max(MIN_ARC_DEG, min(MAX_ARC_DEG, raw))


# Load the input CSV and validate expected columns
df = pd.read_csv(DATA_PATH + INPUT)
if not {'Band', 'Object_Type', 'Count'}.issubset(df.columns):
    raise ValueError('CSV must have columns: Band, Object_Type, Count')

df['Count'] = pd.to_numeric(df['Count'], errors='coerce').fillna(0)
df = df.dropna(subset=['Band', 'Object_Type'])
df = df[df['Count'] > 0]

# Preserve the band order as listed
bands = list(df['Band'].drop_duplicates())

# Group rocket bodies into the debris bucket
obj = df['Object_Type'].str.lower()
is_debris = obj.str.contains('debris') | obj.str.contains('rocket_bodies') | obj.str.contains('r/b')
df['Group'] = np.where(is_debris, 'Debris', 'Satellites')
group_order = ['Debris', 'Satellites']

# Aggregate counts by band and group
agg = df.groupby(['Band', 'Group'], as_index=False)['Count'].sum()
totals = agg.groupby('Band', as_index=False)['Count'].sum().rename(columns={'Count': 'Total'})
merged = agg.merge(totals, on='Band', how='left')
merged['Share'] = np.where(merged['Total'] > 0, merged['Count']/merged['Total'], 0.0)

band_total_map = {b: float(totals.loc[totals['Band'] == b, 'Total'].values[0]) for b in bands}
grand_total = int(sum(band_total_map.values()))
vals = np.array([band_total_map[b] for b in bands], dtype=float)
vmax = float(vals.max()) if len(vals) else 1.0

# Helper for scaling ring thickness
def scale_value(v):
    if vmax <= 0:
      return 0.0
    x = v / vmax
    if SCALING_MODE == 'power':
      return x ** POWER_EXP
    if SCALING_MODE == 'linear':
      return x
    if SCALING_MODE == 'log':
      return np.log1p(v) / np.log1p(vmax)
    return x ** POWER_EXP

#------------------------------------------------------------------------------
# Draw text along a circular arc centered directly above the appropriate ring.
# Text expands in either direction from the centerpoint so too small an
# arc will squish it together. This was literally the hardest part of this
# graph, please no break.
#------------------------------------------------------------------------------
def curved_label_simple(ax, text, r, theta_center_deg, arc_deg,fp=LABEL_FP,
                        color='black', outline=True):
    text = str(text)
    if not text:
        return

    # Set up the arc geometry for the label
    theta_center = math.radians(theta_center_deg)
    half_span = math.radians(max(5.0, arc_deg) / 2.0) * 0.92
    theta1 = theta_center - half_span
    theta2 = theta_center + half_span

    # Guard against mirroring because labels otherwise flip
    x1 = r * math.cos(theta1)
    x2 = r * math.cos(theta2)
    if x1 <= x2:
        start, end, step_sign = theta1, theta2, +1.0   # Step counterclockwise in this branch (CCW)
    else:
        start, end, step_sign = theta2, theta1, -1.0   # Step clockwise otherwise (CW)

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
        rot = math.degrees(ang) - 90.0  # Rotate each character to follow the tangent
        ax.text(x, y, ch, ha='center', va='center',
                rotation=rot, rotation_mode='anchor',
                fontproperties=fp, color=color,
                path_effects=(TEXT_OUTLINE if outline else None))

# Draw the concentric rings
fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw=dict(aspect='equal'),
                       facecolor=BG_COL)
ax.set_axis_off()

current_radius = BASE_RADIUS
max_outer_radius = current_radius
ring_bounds = []  # Store band boundaries as dicts with inner/outer radii

for band in bands:
    total = band_total_map.get(band, 0.0)
    if total <= 0:
        continue

    w = float(max(0.0, min(1.0, scale_value(total))))
    thickness = THICKNESS_MIN + THICKNESS_MAX_EXTRA * w
    inner_r, outer_r = current_radius, current_radius + thickness

    # Split shares between debris and satellites
    bdf = merged[merged['Band'] == band].set_index('Group')
    shares = [float(bdf['Share'].get(g, 0.0)) for g in group_order]

    # Draw annular sectors for each group
    start = START_ANGLE_DEG % 360.0
    for share, g in zip(shares, group_order):
        sweep = 360.0 * share
        if sweep <= 0:
            continue
        theta1, theta2 = start, (start + sweep) % 360.0
        wedge = Wedge((0,0), r=outer_r, theta1=theta1, theta2=theta2,
                      width=(outer_r - inner_r),
                      facecolor=COLOR_MAP.get(g, None),
                      edgecolor='white', linewidth=1.0,
                      alpha=ALPHA, label=g)
        ax.add_patch(wedge)
        start = (start + sweep) % 360.0

    ring_bounds.append({'band': band, 'inner': inner_r, 'outer': outer_r})
    current_radius = outer_r + RING_SPACING
    max_outer_radius = max(max_outer_radius, outer_r)

# Center curved labels above each band
TOP_ANGLE   = 90
OUTER_BLEED = 0.60

for idx, rb in enumerate(ring_bounds):
    band = rb['band']

    # Compute the gap radius
    if idx == 0:
        next_inner = ring_bounds[1]['inner'] if len(ring_bounds) > 1 else rb['outer'] + RING_SPACING - 0.2
        r_label = (rb['outer'] + next_inner - 0.2) / 2.0
    elif idx < len(ring_bounds) - 1:
        next_inner = ring_bounds[idx + 1]['inner']
        r_label = (rb['outer'] + next_inner - 0.2) / 2.0
    else:
        r_label = rb['outer'] + OUTER_BLEED * (RING_SPACING-0.2)  # Add extra bleed on the outermost band

    # Auto-compute the arc for this band at this radius
    arc_deg = compute_arc_deg(band, r_label)

    curved_label_simple(
        ax, band, r=r_label,
        theta_center_deg=TOP_ANGLE,
        arc_deg=arc_deg,
        fp=LABEL_FP, color='black', outline=True
    )


# Handle framing, legend, and title here
R = max_outer_radius + MARGIN
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)

# Format the legend for clarity
handles, labels = ax.get_legend_handles_labels()
seen, handles_clean, labels_clean = set(), [], []
for handle, label in zip(handles, labels):
    if label and label not in seen:
        seen.add(label)
        handles_clean.append(handle)
        labels_clean.append(label)
if handles_clean:
    leg = ax.legend(handles_clean, labels_clean, loc='lower right', frameon=False, title='Object Type')
    plt.setp(leg.get_texts(), fontsize=LEGEND_FONT_SIZE, color='white')
    plt.setp(leg.get_title(), fontsize=LEGEND_TITLE_SIZE, color='white')

plt.title(
    'Distribution of Satellites and Debris Across Orbital Bands',
    pad=20, **TITLE_FONT
)

# Keep export calls commented until needed:
# plt.savefig(VIZ_PATH + OUT_PNG, dpi=1200, facecolor=fig.get_facecolor())
# plt.savefig(VIZ_PATH + OUT_PNG_T, dpi=1200, transparent=True)
# print('Saved:', OUT_PNG, OUT_PNG_T)
