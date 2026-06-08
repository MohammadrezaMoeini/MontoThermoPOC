"""
verification.py — Convergence studies for TransientThermalSolver.

Two studies are performed:
  1. n_walks convergence : fix dt, sweep n_walks.
     MC error scales as 1/√n_walks → T should converge and variance should shrink.
  2. dt convergence      : fix n_walks, sweep dt.
     Backward Euler is O(dt) → T should converge linearly as dt → 0.

System response quantity (QoI): temperature T at a chosen validation point,
t_query seconds AFTER the nozzle deposits material at that point.

Default target: Point #1 (layer 1, bead x=-4.0 mm), t_query = 0.5 s.
Change TARGET_POINT and T_QUERY at the bottom of the file to study any other point.

Run:
    source ~/.MontoThermoPOC312/bin/activate
    python experimentDev/example07_ValidationWithAnton/verification.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.gcode.parser import GCodeParser
from src.gcode.geometry import GeometryBuilder
from solver.transient_thermal_solver import TransientThermalSolver
from solver.config_loader import load_config

HERE   = os.path.dirname(os.path.abspath(__file__))
GCODE  = os.path.join(HERE, 'geometry.gcode')
CONFIG = os.path.join(HERE, '..', '..', 'configs', 'anton_plate.json')

MIN_BEAD_LENGTH = 1.0  # mm — drops lateral x-steps

# ── Validation point coordinates in GCode space ───────────────────────────────
# Transform: x_gcode = x_paper − 4.4,  y_gcode = y_paper − 29.52
# z = layer_height / 2 = 0.2 mm  (midpoint of the 0.4 mm layer)
#
#   Pts 1,2,3  → bead at x = -4.0 mm
#   Pts 4,5,6  → bead at x = -0.4 mm
#   Pts 7,8,9  → bead at x = +4.4 mm

_Z = 0.2   # mm

QUERY_POINTS = {
    1: np.array([[-4.0, -29.12, _Z]], dtype=np.float32),
    2: np.array([[-4.0,   0.08, _Z]], dtype=np.float32),
    3: np.array([[-4.0,  14.48, _Z]], dtype=np.float32),
    4: np.array([[-0.4, -29.12, _Z]], dtype=np.float32),
    5: np.array([[-0.4,   0.08, _Z]], dtype=np.float32),
    6: np.array([[-0.4,  14.48, _Z]], dtype=np.float32),
    7: np.array([[ 4.4, -29.12, _Z]], dtype=np.float32),
    8: np.array([[ 4.4,   0.08, _Z]], dtype=np.float32),
    9: np.array([[ 4.4,  14.48, _Z]], dtype=np.float32),
}

BEAD_X = {1: -4.0, 2: -4.0, 3: -4.0,
           4: -0.4, 5: -0.4, 6: -0.4,
           7:  4.4, 8:  4.4, 9:  4.4}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_moves():
    """Parse the GCode once; return list of long layer-0 moves."""
    all_moves = GCodeParser().parse(GCODE)
    return [m for m in all_moves
            if m['layer'] == 0
            and float(np.hypot(m['x2'] - m['x1'], m['y2'] - m['y1'])) >= MIN_BEAD_LENGTH]


def _find_bead(bead_x, moves):
    """Return the first move whose x-centre matches bead_x (rounded to 1 dp)."""
    for m in moves:
        xc = round((m['x1'] + m['x2']) / 2.0, 1)
        if xc == bead_x:
            return m
    raise ValueError(f'No bead found with x-centre {bead_x} mm')


def _build_mesh(move, cfg):
    gb = GeometryBuilder(nozzle_diameter=cfg['nozzle_diameter'],
                         layer_height=cfg['layer_height'])
    gb.add_move(move)
    verts, faces = gb.get_mesh()
    return verts.astype(np.float32), faces.astype(np.int32)


# ── Core simulation function ──────────────────────────────────────────────────

def simulate_point(pt_num, t_query, n_walks, dt, cfg, move, verts, faces):
    """
    Run the deposition simulation for the bead containing pt_num and return
    T [°C] interpolated at t_query seconds after the nozzle reaches that point.

    Parameters
    ----------
    pt_num  : int          Validation point number (1–9).
    t_query : float        Time [s] after the nozzle deposits at this point.
    n_walks : int          MC walks per query.
    dt      : float        Time step [s].
    cfg     : dict         Loaded config dict.
    move    : dict         GCode move for the bead.
    verts   : np.ndarray   Pre-built bead mesh vertices.
    faces   : np.ndarray   Pre-built bead mesh faces.

    Returns
    -------
    float   Temperature [°C] at t_query after nozzle arrival.
    """
    qpt = QUERY_POINTS[pt_num]

    bead_len = float(np.hypot(move['x2'] - move['x1'], move['y2'] - move['y1']))
    dt_dep   = bead_len / (move['f'] / 60.0)   # total bead traversal time [s]

    # Fraction of bead length to reach this query point
    frac = float(np.clip(
        (qpt[0, 1] - move['y1']) / (move['y2'] - move['y1']), 0.0, 1.0))

    # Number of deposition steps: cover full bead + t_query buffer after nozzle
    n_dep_full  = max(1, int(np.ceil(dt_dep / dt)))
    k0_estimate = max(0, int(np.ceil(frac * n_dep_full)) - 1)
    n_dep = k0_estimate + int(np.ceil(t_query / dt)) + 3

    solver = TransientThermalSolver(
        T_bed     = cfg['T_bed'],
        T_nozzle  = cfg['T_nozzle'],
        T_ambient = cfg['T_ambient'],
        h         = cfg['h'],
        k_cond    = cfg['k_cond'],
        rho       = cfg['rho'],
        cp        = cfg['cp'],
        n_walks   = n_walks,
        dt        = dt,
        grid_shape= cfg['grid_shape'])

    dep = solver.solve_incremental_deposition(
        verts, faces, n_dep,
        bead_y1    = float(move['y1']),
        bead_y2    = float(move['y2']),
        T_substrate= cfg['T_ambient'],
        query_pts  = qpt)

    # Time axis relative to nozzle arrival at this point.
    # At step k0 the nozzle deposits (sets T_nozzle), then the PDE is solved.
    # dep['T'][k0, 0] is the temperature after 1*dt of cooling from T_nozzle.
    k0 = int(dep['nozzle_step'][0])
    T_trace = dep['T'][k0:, 0]                       # (n_dep - k0,) array
    t_trace = np.arange(1, len(T_trace) + 1) * dt    # dt, 2*dt, 3*dt, ...

    if t_query <= 0.0:
        return float(cfg['T_nozzle'])
    if t_query >= t_trace[-1]:
        return float(T_trace[-1])
    return float(np.interp(t_query, t_trace, T_trace))


# ── FEM reference value ───────────────────────────────────────────────────────

def _fem_ref(pt_num, t_query):
    """Interpolate FEM curve at t_query seconds after nozzle arrival at pt_num."""
    from validation import FEM_DATA
    t0 = FEM_DATA[pt_num]['t'][0]
    return float(np.interp(t0 + t_query,
                            FEM_DATA[pt_num]['t'],
                            FEM_DATA[pt_num]['T'],
                            left=FEM_DATA[pt_num]['T'][0],
                            right=FEM_DATA[pt_num]['T'][-1]))


def _exp_ref(pt_num, t_query):
    """Interpolate experimental curve at t_query seconds after nozzle arrival at pt_num."""
    from validation import EXP_DATA
    t0 = EXP_DATA[pt_num]['t'][0]
    return float(np.interp(t0 + t_query,
                            EXP_DATA[pt_num]['t'],
                            EXP_DATA[pt_num]['T'],
                            left=EXP_DATA[pt_num]['T'][0],
                            right=EXP_DATA[pt_num]['T'][-1]))


# ── Study 1: n_walks convergence ──────────────────────────────────────────────

def study_n_walks(pt_num=1,
                  t_query=0.5,
                  n_walks_list=(8, 16, 32, 64, 128, 256, 512, 1024),
                  dt_ref=0.25,
                  n_repeats=5):
    """
    Sweep n_walks with dt fixed; run n_repeats independent realisations at each
    value to expose MC noise.  Plots T vs n_walks with individual dots and the
    FEM reference as a horizontal line.

    Parameters
    ----------
    pt_num       : int   Validation point (1–9).
    t_query      : float Time [s] after nozzle arrival.
    n_walks_list : seq   n_walks values to sweep.
    dt_ref       : float Fixed dt [s].
    n_repeats    : int   Independent MC repeats per n_walks value.
    """
    cfg   = load_config(CONFIG)
    moves = _load_moves()
    move  = _find_bead(BEAD_X[pt_num], moves)
    verts, faces = _build_mesh(move, cfg)

    fem_T = _fem_ref(pt_num, t_query)
    exp_T = _exp_ref(pt_num, t_query)
    all_samples = []   # list of lists, one per n_walks value
    means = []

    print(f'\n── n_walks convergence  (pt#{pt_num},  t_query={t_query} s,  dt={dt_ref} s) ──')
    print(f'   FEM reference at t_query: {fem_T:.1f} °C')
    print(f'   Exp reference at t_query: {exp_T:.1f} °C')
    for nw in n_walks_list:
        samples = [
            simulate_point(pt_num, t_query, nw, dt_ref, cfg, move, verts, faces)
            for _ in range(n_repeats)
        ]
        all_samples.append(samples)
        means.append(float(np.mean(samples)))
        print(f'  n_walks={nw:5d}   T = {np.mean(samples):.2f} ± {np.std(samples):.2f} °C')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # Individual repeat dots
    for i, (nw, samples) in enumerate(zip(n_walks_list, all_samples)):
        ax.scatter([nw] * len(samples), samples,
                   color='steelblue', alpha=0.5, s=30, zorder=3)

    # Mean line
    ax.plot(n_walks_list, means, 'o-', color='steelblue', lw=1.8,
            label=f'MC WoS mean  ({n_repeats} repeats each)')

    # Reference lines
    ax.axhline(fem_T, ls='--', color='crimson', lw=1.5,
               label=f'FEM (Trofimov 2022)  = {fem_T:.1f} °C')
    ax.axhline(exp_T, ls='--', color='black', lw=1.5,
               label=f'Experimental (IR)  = {exp_T:.1f} °C')

    ax.set_xlabel('n_walks', fontsize=11)
    ax.set_ylabel('Temperature  [°C]', fontsize=11)
    ax.set_title(
        f'T vs n_walks — Pt #{pt_num},  t = {t_query} s after nozzle,  dt = {dt_ref} s',
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(HERE, f'convergence_nwalks_pt{pt_num:02d}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close(fig)


# ── Study 2: dt convergence ───────────────────────────────────────────────────

def study_dt(pt_num=1,
             t_query=0.5,
             dt_list=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125),
             n_walks_ref=512):
    """
    Sweep dt with n_walks fixed.  Plots T vs dt and the FEM reference.

    Parameters
    ----------
    pt_num      : int   Validation point (1–9).
    t_query     : float Time [s] after nozzle arrival.
    dt_list     : seq   dt values [s], coarse to fine.
    n_walks_ref : int   Fixed n_walks (large enough to suppress MC noise).
    """
    cfg   = load_config(CONFIG)
    moves = _load_moves()
    move  = _find_bead(BEAD_X[pt_num], moves)
    verts, faces = _build_mesh(move, cfg)

    fem_T  = _fem_ref(pt_num, t_query)
    exp_T  = _exp_ref(pt_num, t_query)
    T_vals = []

    print(f'\n── dt convergence  (pt#{pt_num},  t_query={t_query} s,  n_walks={n_walks_ref}) ──')
    print(f'   FEM reference at t_query: {fem_T:.1f} °C')
    print(f'   Exp reference at t_query: {exp_T:.1f} °C')
    for dt in dt_list:
        T = simulate_point(pt_num, t_query, n_walks_ref, dt, cfg, move, verts, faces)
        T_vals.append(T)
        print(f'  dt={dt:.4f} s   T = {T:.2f} °C')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(dt_list, T_vals, 'o-', color='steelblue', lw=1.8,
            label='MC WoS')

    ax.axhline(fem_T, ls='--', color='crimson', lw=1.5,
               label=f'FEM (Trofimov 2022)  = {fem_T:.1f} °C')
    ax.axhline(exp_T, ls='--', color='black', lw=1.5,
               label=f'Experimental (IR)  = {exp_T:.1f} °C')

    ax.set_xlabel('dt  [s]', fontsize=11)
    ax.set_ylabel('Temperature  [°C]', fontsize=11)
    ax.set_title(
        f'T vs dt — Pt #{pt_num},  t = {t_query} s after nozzle,  n_walks = {n_walks_ref}',
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(HERE, f'convergence_dt_pt{pt_num:02d}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TARGET_POINT = 1      # change to any point 1–9
    T_QUERY      = 0.5    # seconds after nozzle reaches that point

    study_n_walks(pt_num=TARGET_POINT, t_query=T_QUERY)
    study_dt(pt_num=TARGET_POINT, t_query=T_QUERY)