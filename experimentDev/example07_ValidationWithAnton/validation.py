"""
Validation against Trofimov 2022 — Figure 11 data.

Plots the 27 experimentally measured T vs time curves AND the corresponding
FEM predictions from:
  Trofimov, A. et al. (2022). Experimentally validated modeling of
  temperature distribution during FFF. tests/Anton/...pdf

Layout: 3×3 grid of subplots
  rows  → layers 1 / 10 / 20
  cols  → x₁ = 0.4 / 4.0 / 8.8 mm (width direction)

Within each subplot three curve pairs show points at x₂ = 0.4, 29.6, 44 mm:
  black  → x₂ = 0.4  mm
  red    → x₂ = 29.6 mm
  blue   → x₂ = 44.0 mm
  solid  → Experimental (IR thermography)
  dashed → FEM (Trofimov 2022 ABAQUS model)

Data source: manually digitised from Figure 11, Trofimov 2022.
Accuracy: ±5–10 °C, ±0.05 s (limited by reading a printed PDF figure).

Later: Monte Carlo (WoS) predictions will be added as dash-dot lines.

Run with:
    source ~/.MontoThermoPOC312/bin/activate
    python experimentDev/example07_ValidationWithAnton/validation.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Point metadata ────────────────────────────────────────────────────────────
#   Layer 1  → points  1–9   (z = 0.2 mm)
#   Layer 10 → points 10–18  (z = 2.0 mm)
#   Layer 20 → points 19–27  (z = 4.0 mm)
#   Coordinates (x₁, x₂, z) in mm

POINT_COORDS = {
     1: (0.4,  0.4, 0.2),   2: (0.4, 29.6, 0.2),   3: (0.4, 44.0, 0.2),
     4: (4.0,  0.4, 0.2),   5: (4.0, 29.6, 0.2),   6: (4.0, 44.0, 0.2),
     7: (8.8,  0.4, 0.2),   8: (8.8, 29.6, 0.2),   9: (8.8, 44.0, 0.2),
    10: (0.4,  0.4, 2.0),  11: (0.4, 29.6, 2.0),  12: (0.4, 44.0, 2.0),
    13: (4.0,  0.4, 2.0),  14: (4.0, 29.6, 2.0),  15: (4.0, 44.0, 2.0),
    16: (8.8,  0.4, 2.0),  17: (8.8, 29.6, 2.0),  18: (8.8, 44.0, 2.0),
    19: (0.4,  0.4, 4.0),  20: (0.4, 29.6, 4.0),  21: (0.4, 44.0, 4.0),
    22: (4.0,  0.4, 4.0),  23: (4.0, 29.6, 4.0),  24: (4.0, 44.0, 4.0),
    25: (8.8,  0.4, 4.0),  26: (8.8, 29.6, 4.0),  27: (8.8, 44.0, 4.0),
}

# ── Experimental data (solid lines in Figure 11) ─────────────────────────────
# Values digitised visually from the PDF; accuracy ≈ ±5–10 °C, ±0.05 s.

EXP_DATA = {

    # ── (a) Layer 1, x₁=0.4, t ≈ 0–2 s  (half-speed layer → ~2 s/bead) ───────
    1: {'t': np.array([0.00, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.20, 1.50, 2.00]),
        'T': np.array([140,  126,  113,   99,   88,   77,   70,   63,   59,   55])},

    2: {'t': np.array([1.00, 1.10, 1.20, 1.35, 1.50, 1.65, 1.80, 2.00]),
        'T': np.array([135,  118,  107,   96,   90,   85,   82,   80])},

    3: {'t': np.array([1.50, 1.60, 1.70, 1.80, 1.90, 2.00]),
        'T': np.array([130,  116,  106,   97,   89,   82])},

    # ── (b) Layer 1, x₁=4.0, t ≈ 23.5–26.5 s ────────────────────────────────
    4: {'t': np.array([23.50, 23.65, 23.80, 24.00, 24.20, 24.40, 24.70, 25.00, 25.50]),
        'T': np.array([155,   133,   115,    98,    85,    75,    67,    63,    58])},

    5: {'t': np.array([25.00, 25.15, 25.30, 25.50, 25.70, 25.90, 26.10]),
        'T': np.array([135,   116,   103,    92,    84,    77,    72])},

    6: {'t': np.array([25.50, 25.65, 25.80, 26.00, 26.20, 26.50]),
        'T': np.array([130,   111,    98,    88,    82,    77])},

    # ── (c) Layer 1, x₁=8.8, t ≈ 43.5–46 s  (ΔT_max ≈ 24 °C at pt 7) ───────
    7: {'t': np.array([43.50, 43.65, 43.80, 44.00, 44.20, 44.40, 44.60, 44.80]),
        'T': np.array([145,   121,   102,    84,    72,    63,    58,    53])},

    8: {'t': np.array([44.80, 44.90, 45.00, 45.10, 45.25, 45.40, 45.60, 45.80]),
        'T': np.array([120,   112,   104,    96,    87,    80,    74,    69])},

    9: {'t': np.array([45.30, 45.45, 45.60, 45.75, 45.90, 46.00]),
        'T': np.array([120,   108,    97,    88,    83,    78])},

    # ── (d) Layer 10, x₁=0.4, t ≈ 259.4–260.6 s ─────────────────────────────
    10: {'t': np.array([259.40, 259.50, 259.60, 259.70, 259.80, 259.90, 260.10, 260.30, 260.60]),
         'T': np.array([150,    143,    136,    130,    125,    120,    115,    110,    105])},

    11: {'t': np.array([259.90, 260.00, 260.10, 260.20, 260.30, 260.40, 260.60]),
         'T': np.array([130,    125,    121,    118,    115,    113,    110])},

    12: {'t': np.array([260.20, 260.30, 260.40, 260.50, 260.60]),
         'T': np.array([130,    125,    122,    118,    115])},

    # ── (e) Layer 10, x₁=4.0, t ≈ 271.4–272.6 s ─────────────────────────────
    13: {'t': np.array([271.40, 271.50, 271.60, 271.70, 271.80, 271.90, 272.10, 272.30, 272.60]),
         'T': np.array([150,    143,    136,    129,    123,    117,    111,    105,     95])},

    14: {'t': np.array([272.00, 272.10, 272.20, 272.30, 272.40, 272.50, 272.60]),
         'T': np.array([140,    135,    130,    126,    122,    118,    114])},

    15: {'t': np.array([272.20, 272.30, 272.40, 272.50, 272.60]),
         'T': np.array([145,    139,    134,    129,    123])},

    # ── (f) Layer 10, x₁=8.8, t ≈ 281.4–282.6 s ─────────────────────────────
    16: {'t': np.array([281.40, 281.50, 281.60, 281.70, 281.80, 281.90, 282.10, 282.30, 282.60]),
         'T': np.array([150,    142,    135,    128,    122,    116,    109,    102,     93])},

    17: {'t': np.array([282.00, 282.10, 282.20, 282.30, 282.40, 282.50, 282.60]),
         'T': np.array([140,    134,    129,    125,    121,    117,    113])},

    18: {'t': np.array([282.20, 282.30, 282.40, 282.50, 282.60]),
         'T': np.array([145,    139,    134,    130,    125])},

    # ── (g) Layer 20, x₁=0.4, t ≈ 467–468.5 s ───────────────────────────────
    19: {'t': np.array([467.00, 467.10, 467.20, 467.30, 467.50, 467.70, 467.90, 468.20, 468.50]),
         'T': np.array([145,    140,    135,    131,    125,    119,    115,    109,    103])},

    20: {'t': np.array([467.50, 467.60, 467.70, 467.80, 467.90, 468.10, 468.30, 468.50]),
         'T': np.array([140,    137,    133,    130,    127,    122,    118,    112])},

    21: {'t': np.array([468.00, 468.10, 468.20, 468.30, 468.40, 468.50]),
         'T': np.array([140,    137,    134,    131,    128,    123])},

    # ── (h) Layer 20, x₁=4.0, t ≈ 478.4–479.6 s ─────────────────────────────
    22: {'t': np.array([478.40, 478.50, 478.60, 478.70, 478.80, 479.00, 479.20, 479.40, 479.60]),
         'T': np.array([145,    138,    132,    126,    121,    113,    107,    102,     97])},

    23: {'t': np.array([479.00, 479.10, 479.20, 479.30, 479.40, 479.50, 479.60]),
         'T': np.array([135,    131,    127,    124,    120,    116,    112])},

    24: {'t': np.array([479.20, 479.30, 479.40, 479.50, 479.60]),
         'T': np.array([140,    136,    132,    128,    124])},

    # ── (i) Layer 20, x₁=8.8, t ≈ 489.2–490.4 s ─────────────────────────────
    25: {'t': np.array([489.20, 489.30, 489.40, 489.50, 489.60, 489.70, 489.80, 490.00, 490.20, 490.40]),
         'T': np.array([140,    133,    127,    121,    115,    110,    106,     99,     94,     90])},

    26: {'t': np.array([490.00, 490.10, 490.20, 490.30, 490.40]),
         'T': np.array([135,    131,    127,    123,    118])},

    27: {'t': np.array([490.20, 490.30, 490.40]),
         'T': np.array([132,    128,    124])},
}

# ── FEM data (dashed lines in Figure 11) ─────────────────────────────────────
# Digitised from the dashed FEM curves in Figure 11 (Trofimov 2022 ABAQUS).
# FEM curves are generally smooth (no measurement noise).
# Key known discrepancy: Point #7 → FEM overestimates by ~24 °C (middle section).
# Points #4 overestimates by ~10–15 °C. All other points within ±5–8 °C.

FEM_DATA = {

    # ── (a) Layer 1, x₁=0.4 — FEM slightly below Exp (≈ −4 °C) ─────────────
    1: {'t': np.array([0.00, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.20, 1.50, 2.00]),
        'T': np.array([138,  122,  109,   95,   84,   73,   66,   59,   56,   52])},

    2: {'t': np.array([1.00, 1.10, 1.20, 1.35, 1.50, 1.65, 1.80, 2.00]),
        'T': np.array([132,  115,  104,   93,   87,   82,   79,   76])},

    3: {'t': np.array([1.50, 1.60, 1.70, 1.80, 1.90, 2.00]),
        'T': np.array([127,  113,  103,   94,   86,   79])},

    # ── (b) Layer 1, x₁=4.0 — FEM #4 overestimates ~10 °C, 5&6 close ────────
    4: {'t': np.array([23.50, 23.65, 23.80, 24.00, 24.20, 24.40, 24.70, 25.00, 25.50]),
        'T': np.array([158,   142,   127,   111,    97,    86,    75,    70,    63])},

    5: {'t': np.array([25.00, 25.15, 25.30, 25.50, 25.70, 25.90, 26.10]),
        'T': np.array([138,   119,   106,    95,    87,    80,    75])},

    6: {'t': np.array([25.50, 25.65, 25.80, 26.00, 26.20, 26.50]),
        'T': np.array([133,   114,   101,    91,    85,    80])},

    # ── (c) Layer 1, x₁=8.8 — FEM #7 overestimates by ~24 °C (middle) ───────
    # Exp #7 cools fast (low bed-contact h_b); FEM misses this → overestimates.
    7: {'t': np.array([43.50, 43.65, 43.80, 44.00, 44.20, 44.40, 44.60, 44.80]),
        'T': np.array([145,   128,   114,   104,    96,    88,    81,    74])},

    8: {'t': np.array([44.80, 44.90, 45.00, 45.10, 45.25, 45.40, 45.60, 45.80]),
        'T': np.array([124,   116,   108,   101,    92,    85,    79,    74])},

    9: {'t': np.array([45.30, 45.45, 45.60, 45.75, 45.90, 46.00]),
        'T': np.array([124,   112,   101,    92,    87,    82])},

    # ── (d) Layer 10, x₁=0.4 — FEM tracks Exp closely (±5 °C) ───────────────
    10: {'t': np.array([259.40, 259.50, 259.60, 259.70, 259.80, 259.90, 260.10, 260.30, 260.60]),
         'T': np.array([153,    147,    141,    135,    130,    125,    119,    114,    109])},

    11: {'t': np.array([259.90, 260.00, 260.10, 260.20, 260.30, 260.40, 260.60]),
         'T': np.array([134,    129,    125,    122,    119,    117,    114])},

    12: {'t': np.array([260.20, 260.30, 260.40, 260.50, 260.60]),
         'T': np.array([134,    129,    126,    123,    119])},

    # ── (e) Layer 10, x₁=4.0 ─────────────────────────────────────────────────
    13: {'t': np.array([271.40, 271.50, 271.60, 271.70, 271.80, 271.90, 272.10, 272.30, 272.60]),
         'T': np.array([154,    148,    141,    135,    128,    122,    116,    110,    101])},

    14: {'t': np.array([272.00, 272.10, 272.20, 272.30, 272.40, 272.50, 272.60]),
         'T': np.array([144,    139,    134,    130,    127,    123,    119])},

    15: {'t': np.array([272.20, 272.30, 272.40, 272.50, 272.60]),
         'T': np.array([149,    143,    138,    134,    128])},

    # ── (f) Layer 10, x₁=8.8 ─────────────────────────────────────────────────
    16: {'t': np.array([281.40, 281.50, 281.60, 281.70, 281.80, 281.90, 282.10, 282.30, 282.60]),
         'T': np.array([154,    147,    140,    133,    127,    121,    114,    107,     98])},

    17: {'t': np.array([282.00, 282.10, 282.20, 282.30, 282.40, 282.50, 282.60]),
         'T': np.array([145,    139,    134,    130,    126,    122,    118])},

    18: {'t': np.array([282.20, 282.30, 282.40, 282.50, 282.60]),
         'T': np.array([150,    144,    139,    135,    130])},

    # ── (g) Layer 20, x₁=0.4 ─────────────────────────────────────────────────
    19: {'t': np.array([467.00, 467.10, 467.20, 467.30, 467.50, 467.70, 467.90, 468.20, 468.50]),
         'T': np.array([148,    143,    138,    134,    128,    123,    119,    113,    107])},

    20: {'t': np.array([467.50, 467.60, 467.70, 467.80, 467.90, 468.10, 468.30, 468.50]),
         'T': np.array([144,    141,    137,    134,    131,    126,    122,    116])},

    21: {'t': np.array([468.00, 468.10, 468.20, 468.30, 468.40, 468.50]),
         'T': np.array([144,    141,    138,    135,    132,    127])},

    # ── (h) Layer 20, x₁=4.0 ─────────────────────────────────────────────────
    22: {'t': np.array([478.40, 478.50, 478.60, 478.70, 478.80, 479.00, 479.20, 479.40, 479.60]),
         'T': np.array([149,    142,    136,    130,    125,    117,    111,    106,    101])},

    23: {'t': np.array([479.00, 479.10, 479.20, 479.30, 479.40, 479.50, 479.60]),
         'T': np.array([139,    135,    131,    128,    124,    120,    116])},

    24: {'t': np.array([479.20, 479.30, 479.40, 479.50, 479.60]),
         'T': np.array([144,    140,    136,    132,    128])},

    # ── (i) Layer 20, x₁=8.8 ─────────────────────────────────────────────────
    25: {'t': np.array([489.20, 489.30, 489.40, 489.50, 489.60, 489.70, 489.80, 490.00, 490.20, 490.40]),
         'T': np.array([144,    137,    131,    125,    120,    115,    111,    104,     99,     95])},

    26: {'t': np.array([490.00, 490.10, 490.20, 490.30, 490.40]),
         'T': np.array([139,    135,    131,    127,    122])},

    27: {'t': np.array([490.20, 490.30, 490.40]),
         'T': np.array([136,    132,    128])},
}

# ── Subplot layout ────────────────────────────────────────────────────────────
# (row, col, letter, layer, x₁_mm, (pt_black, pt_red, pt_blue))

SUBPLOTS = [
    (0, 0, '(a)',  1, 0.4, ( 1,  2,  3)),
    (0, 1, '(b)',  1, 4.0, ( 4,  5,  6)),
    (0, 2, '(c)',  1, 8.8, ( 7,  8,  9)),
    (1, 0, '(d)', 10, 0.4, (10, 11, 12)),
    (1, 1, '(e)', 10, 4.0, (13, 14, 15)),
    (1, 2, '(f)', 10, 8.8, (16, 17, 18)),
    (2, 0, '(g)', 20, 0.4, (19, 20, 21)),
    (2, 1, '(h)', 20, 4.0, (22, 23, 24)),
    (2, 2, '(i)', 20, 8.8, (25, 26, 27)),
]

COLORS = ['black', 'red', 'steelblue']   # x₂ = 0.4, 29.6, 44.0 mm

# ── Monte Carlo results (layer 1 only, points #1–9) ───────────────────────────
# Loaded from mc_results_layer1.pkl produced by example07.py.
# Time is re-aligned to the paper's absolute print-time reference:
#   paper t=0 aligns with the start of the first bead (pt#1 bead start).
#   Each bead group's MC start time is shifted to match the paper's bead start.
#
#   Bead for pts#1-3: paper bead start = EXP_DATA[1]['t'][0] = 0.00 s
#   Bead for pts#4-6: paper bead start = EXP_DATA[4]['t'][0] = 23.50 s
#   Bead for pts#7-9: paper bead start = EXP_DATA[7]['t'][0] = 43.50 s

MC_DATA = {}
_mc_pkl = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mc_results_layer1.pkl')
if os.path.exists(_mc_pkl):
    with open(_mc_pkl, 'rb') as _f:
        _raw = pickle.load(_f)
    _bead_starts_mc    = {1: _raw[1]['t'][0], 4: _raw[4]['t'][0], 7: _raw[7]['t'][0]}
    _bead_starts_paper = {1: EXP_DATA[1]['t'][0],
                          4: EXP_DATA[4]['t'][0],
                          7: EXP_DATA[7]['t'][0]}
    _bead_key = {1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4, 7: 7, 8: 7, 9: 7}
    for pt_num in range(1, 10):
        bk = _bead_key[pt_num]
        shift = _bead_starts_mc[bk] - _bead_starts_paper[bk]
        MC_DATA[pt_num] = {
            't': _raw[pt_num]['t'] - shift,
            'T': _raw[pt_num]['T'],
        }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_figure11():
    mc_available = bool(MC_DATA)
    mc_label = 'Monte Carlo WoS (example07)' if mc_available else 'Monte Carlo WoS (not yet run)'
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(
        'Figure 11 reproduction — Trofimov 2022\n'
        'Solid = Experimental (IR)   |   Dashed = FEM (ABAQUS)   |'
        f'   Dash-dot = {mc_label}',
        fontsize=10, y=0.995)

    for row, col, letter, layer, x1, pts in SUBPLOTS:
        ax = axes[row, col]

        for pt_num, color in zip(pts, COLORS):
            x1c, x2c, zc = POINT_COORDS[pt_num]
            lbl = f'#{pt_num}  x₂={x2c:.0f}mm'

            # Experimental — solid line with small markers
            exp = EXP_DATA[pt_num]
            ax.plot(exp['t'], exp['T'],
                    color=color, lw=1.8, ls='-', marker='o', ms=3,
                    label=f'Exp {lbl}')

            # FEM — dashed, same colour, no markers
            fem = FEM_DATA[pt_num]
            ax.plot(fem['t'], fem['T'],
                    color=color, lw=1.5, ls='--',
                    label=f'FEM {lbl}')

            # Monte Carlo WoS predictions (layer 1 only, pts #1–9)
            if mc_available and pt_num in MC_DATA:
                mc = MC_DATA[pt_num]
                ax.plot(mc['t'], mc['T'],
                        color=color, lw=1.5, ls='-.',
                        label=f'MC {lbl}')

        # Annotate the known 24 °C discrepancy on subplot (c)
        if letter == '(c)':
            ax.annotate('ΔT_max ≈ 24 °C\n(Pt #7)',
                        xy=(44.3, 88), xytext=(44.55, 105),
                        fontsize=7, color='black',
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

        ax.set_title(f'{letter}  Layer {layer},  x₁ = {x1:.1f} mm', fontsize=9)
        ax.set_xlabel('Time  [s]', fontsize=8)
        ax.set_ylabel('Temperature  [°C]', fontsize=8)
        ax.set_ylim(20, 170)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='upper right', ncol=1)
        ax.grid(True, alpha=0.25)

    # Shared legend for line styles
    leg_exp = mlines.Line2D([], [], color='gray', ls='-',  lw=1.5, label='Experimental (IR)')
    leg_fem = mlines.Line2D([], [], color='gray', ls='--', lw=1.5, label='FEM (Trofimov 2022)')
    leg_mc  = mlines.Line2D([], [], color='gray', ls='-.', lw=1.5, label=mc_label)
    fig.legend(handles=[leg_exp, leg_fem, leg_mc],
               loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'T_validation_exp_fem_mc.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.show()


if __name__ == '__main__':
    plot_figure11()