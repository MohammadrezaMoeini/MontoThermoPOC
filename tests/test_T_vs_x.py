"""
Plot T(x) at bead midpoints for time steps t=0, t=5, t=9.

For each time step t, queries T at the midpoint of every bead deposited
so far (0 .. t).  Each bead is solved as a single watertight cuboid.
The x-axis shows the x-coordinate of each bead midpoint.

Run:
    python tests/test_T_vs_x.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.gcode.parser import GCodeParser
from src.gcode.geometry import GeometryBuilder
from solver.thermal_solver import ThermalSolver

TESTS_DIR    = Path(__file__).parent
GCODE_FILE   = TESTS_DIR / "20mmbox.gcode"
PLOT_STEPS   = {0, 5, 9}
LAYER_HEIGHT = 0.2
NOZZLE_DIA   = 0.4
N_WALKS      = 512

CUM_TIMES = {0: 0.075, 5: 1.070, 9: 1.990}
COLORS    = {0: 'steelblue', 5: 'darkorange', 9: 'crimson'}


def single_bead_mesh(move):
    gb = GeometryBuilder(nozzle_diameter=NOZZLE_DIA, layer_height=LAYER_HEIGHT)
    gb.add_move(move)
    return gb.get_mesh()


def bead_midpoint(move):
    return np.array([
        (move['x1'] + move['x2']) / 2.0,
        (move['y1'] + move['y2']) / 2.0,
        move['z'] - LAYER_HEIGHT / 2.0,
    ], dtype=np.float32)


def main():
    # collect valid moves
    all_moves = GCodeParser().parse(str(GCODE_FILE))
    gb_check  = GeometryBuilder(nozzle_diameter=NOZZLE_DIA,
                                layer_height=LAYER_HEIGHT)
    valid_moves = []
    for move in all_moves:
        prev = gb_check._vertex_count
        gb_check.add_move(move)
        if gb_check._vertex_count > prev:
            valid_moves.append(move)
        if len(valid_moves) > max(PLOT_STEPS):
            break

    solver  = ThermalSolver(T_bed=60.0, T_nozzle=200.0, T_ambient=20.0,
                            n_walks=N_WALKS)
    results = {}   # t_idx -> (xs, Ts)

    for t_idx in sorted(PLOT_STEPS):
        print(f"\n=== time step t={t_idx} ===")
        xs, Ts = [], []

        for bead_i in range(t_idx + 1):
            move            = valid_moves[bead_i]
            verts, faces    = single_bead_mesh(move)
            qpt             = bead_midpoint(move)
            is_current_bead = (bead_i == t_idx)

            T = solver.solve(verts, faces, qpt,
                             is_current_bead=is_current_bead)
            xs.append(float(qpt[0]))
            Ts.append(T)
            print(f"  bead={bead_i}  x={qpt[0]:.1f}  y={qpt[1]:.1f}"
                  f"  z={qpt[2]:.3f}  current={is_current_bead}"
                  f"  T={T:.1f}°C", flush=True)

        results[t_idx] = (np.array(xs), np.array(Ts))
        print(f"--- t={t_idx} done ---")

    # ---- plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    for t_idx, (xs, Ts) in sorted(results.items()):
        cum_t = CUM_TIMES[t_idx]
        n_beads = t_idx + 1
        ax.plot(xs, Ts, 'o-', color=COLORS[t_idx], lw=2, ms=8,
                label=f't={t_idx}  (t={cum_t:.3f}s,  {n_beads} beads)')

    ax.axhline(60,  color='cyan',   lw=1.2, linestyle='--', alpha=0.8,
               label='T_bed = 60°C')
    ax.axhline(200, color='gold',   lw=1.2, linestyle='--', alpha=0.8,
               label='T_nozzle = 200°C')
    ax.axhline(20,  color='silver', lw=1.2, linestyle=':',  alpha=0.8,
               label='T_ambient = 20°C')

    ax.set_xlabel('X coordinate of bead midpoint (mm)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(
        'T at bead midpoints vs X position\n'
        'per-bead quasi-static solution at t=0, t=5, t=9',
        fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_ylim(0, 220)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()