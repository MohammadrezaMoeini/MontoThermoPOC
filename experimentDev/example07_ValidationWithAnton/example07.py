"""
Example 07: Validation against Trofimov 2022 — regular plate, layer 1.

Runs the transient thermal solver on the Anton GCode (geometry a) for all
23 beads in layer 1 and compares predicted T vs time at the 9 layer-1
validation points (paper points #1–9) against the Exp and FEM curves in
validation.py.

Run with:
    source ~/.MontoThermoPOC312/bin/activate
    python experimentDev/example07_ValidationWithAnton/example07.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.gcode.parser import GCodeParser
from src.gcode.geometry import GeometryBuilder
from solver.transient_thermal_solver import TransientThermalSolver
from solver.config_loader import load_config

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(__file__)
GCODE  = os.path.join(HERE, 'geometry.gcode')
CONFIG = os.path.join(HERE, '..', '..', 'configs', 'anton_plate.json')

# ── Step 1: Parse + filter ────────────────────────────────────────────────────

MIN_BEAD_LENGTH = 1.0   # mm — drops lateral x-steps (0.4 mm)

all_moves = GCodeParser().parse(GCODE)

def bead_length(m):
    return float(np.hypot(m['x2'] - m['x1'], m['y2'] - m['y1']))

layer1_beads = [
    m for m in all_moves
    if m['layer'] == 0 and bead_length(m) >= MIN_BEAD_LENGTH
]

print(f"Parsed {len(all_moves)} moves  →  {len(layer1_beads)} layer-1 beads after filter")

# ── Step 2: Setup solver from config ─────────────────────────────────────────

cfg = load_config(CONFIG)

solver = TransientThermalSolver(
    T_bed     = cfg['T_bed'],
    T_nozzle  = cfg['T_nozzle'],
    T_ambient = cfg['T_ambient'],
    h         = cfg['h'],
    k_cond    = cfg['k_cond'],
    rho       = cfg['rho'],
    cp        = cfg['cp'],
    n_walks   = cfg['n_walks'],
    dt        = cfg['dt'],
    grid_shape= cfg['grid_shape'])

print(f"\nSolver  T_bed={cfg['T_bed']}°C  T_nozzle={cfg['T_nozzle']}°C  "
      f"h={cfg['h']} W/m²K  k={cfg['k_cond']} W/mK")
print(f"        α={solver.alpha:.4f} mm²/s  σ={solver.sigma:.4f} mm⁻²  dt={cfg['dt']}s")

# ── Validation query points in GCode space ────────────────────────────────────
#
# Transform: x_gcode = x_paper − 4.4,  y_gcode = y_paper − 29.52
# z = layer_height / 2 = 0.2 mm  (midpoint of first 0.4 mm layer)
#
# Bead x=-4.0 → paper points #1, #2, #3
# Bead x=-0.4 → paper points #4, #5, #6
# Bead x=+4.4 → paper points #7, #8, #9

Z_VAL = cfg['layer_height'] / 2.0   # 0.2 mm — midpoint of layer 1

VAL_BEADS = {
    -4.0: {'pts': np.array([[-4.0, -29.12, Z_VAL],   # pt #1
                             [-4.0,   0.08, Z_VAL],   # pt #2
                             [-4.0,  14.48, Z_VAL]],  # pt #3
                            dtype=np.float32),
           'pt_nums': [1, 2, 3]},

    -0.4: {'pts': np.array([[-0.4, -29.12, Z_VAL],   # pt #4
                             [-0.4,   0.08, Z_VAL],   # pt #5
                             [-0.4,  14.48, Z_VAL]],  # pt #6
                            dtype=np.float32),
           'pt_nums': [4, 5, 6]},

     4.4: {'pts': np.array([[ 4.4, -29.12, Z_VAL],   # pt #7
                             [ 4.4,   0.08, Z_VAL],   # pt #8
                             [ 4.4,  14.48, Z_VAL]],  # pt #9
                            dtype=np.float32),
           'pt_nums': [7, 8, 9]},
}

# ── Simulate all 23 layer-1 beads ─────────────────────────────────────────────

gb      = GeometryBuilder(nozzle_diameter=cfg['nozzle_diameter'],
                          layer_height=cfg['layer_height'])
T_ic    = cfg['T_ambient']   # carried between beads
t_global = 0.0
val_results = {}             # pt_num → {'t': array, 'T': array}

print(f"\nSimulating {len(layer1_beads)} beads  "
      f"(n_walks={cfg['n_walks']}, dt={cfg['dt']}s, "
      f"n_cooling={cfg['n_cooling_steps']})...\n")

for i, move in enumerate(layer1_beads):

    # Build bead mesh
    gb.reset()
    gb.add_move(move)
    verts, faces = gb.get_mesh()
    if len(verts) == 0:
        continue
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int32)

    # Identify validation bead by x-centre (rounded to 1 dp)
    bead_x = round((move['x1'] + move['x2']) / 2.0, 1)
    if bead_x in VAL_BEADS:
        qpts    = VAL_BEADS[bead_x]['pts']
        pt_nums = VAL_BEADS[bead_x]['pt_nums']
        is_val  = True
    else:
        # Single midpoint — only needed for T_ic carryover
        qpts = np.array([[bead_x,
                          (move['y1'] + move['y2']) / 2.0,
                          Z_VAL]], dtype=np.float32)
        pt_nums = []
        is_val  = False

    # Deposition steps
    length = bead_length(move)
    dt_dep = length / (move['f'] / 60.0)
    n_dep  = max(1, round(dt_dep / solver.dt))

    tag = f"VAL pts={pt_nums}" if is_val else "   "
    print(f"  bead {i:2d}  x={bead_x:+.1f}  {tag}", end='  ', flush=True)

    t_bead_start = t_global   # absolute time at start of this bead

    if is_val:
        # --- Incremental deposition (moving nozzle) for validation beads ---
        # Each query point starts at T_nozzle the moment the nozzle reaches it;
        # already-deposited material continues cooling naturally each step.
        dep = solver.solve_incremental_deposition(
            verts, faces, n_dep,
            bead_y1=float(move['y1']), bead_y2=float(move['y2']),
            T_substrate=T_ic,
            query_pts=qpts)
        T_initial_cool = dep['T_grid_final']
    else:
        # --- Whole-bead deposition for non-validation beads ---
        res_dep = solver.solve_transient(
            verts, faces, n_dep,
            is_current_bead_schedule=[True] * n_dep,
            T_initial=cfg['T_nozzle'],
            query_pts=qpts)
        T_initial_cool = float(res_dep['T'][-1].mean())

    # --- Cooling phase (common to both paths) ---
    res_cool = solver.solve_transient(
        verts, faces, cfg['n_cooling_steps'],
        is_current_bead_schedule=[False] * cfg['n_cooling_steps'],
        T_initial=T_initial_cool,
        query_pts=qpts)

    t_cool_abs = (t_bead_start + n_dep * solver.dt) + res_cool['times']

    # Update global time and carry-forward IC
    t_global += (n_dep + cfg['n_cooling_steps']) * solver.dt
    T_ic      = float(res_cool['T'][-1].mean())

    print(f"T_ic→{T_ic:.1f}°C")

    # Store per-point T history for validation beads
    if is_val:
        for j, pt_num in enumerate(pt_nums):
            k0 = int(dep['nozzle_step'][j])
            # Deposition history from the step the nozzle reached this point
            t_dep_j = t_bead_start + dep['times'][k0 + 1:]
            T_dep_j = dep['T'][k0 + 1:, j]
            # Cooling history (skip duplicate at t=0 of cooling)
            t_cool_j = t_cool_abs[1:]
            T_cool_j = res_cool['T'][1:, j]
            val_results[pt_num] = {
                't': np.concatenate([t_dep_j, t_cool_j]),
                'T': np.concatenate([T_dep_j, T_cool_j]),
            }

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\nDone. Captured T history for points: {sorted(val_results.keys())}")
print()
for pt_num in sorted(val_results.keys()):
    d = val_results[pt_num]
    print(f"  Pt #{pt_num:2d}  t=[{d['t'][0]:.2f}, {d['t'][-1]:.2f}]s  "
          f"T=[{d['T'].min():.1f}, {d['T'].max():.1f}]°C")

# Save for plotting
import pickle
out = os.path.join(HERE, 'mc_results_layer1.pkl')
with open(out, 'wb') as f:
    pickle.dump(val_results, f)
print(f"\nResults saved → {out}")