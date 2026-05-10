"""
Quasi-static thermal simulation over 10 time steps.

At each time step t_i:
  - Bead i is added to the accumulated geometry (GeometryBuilder)
  - The steady-state heat equation is solved on the full accumulated mesh
  - Temperature is queried at the midpoint of the newly deposited bead
  - Result: T(x_mid, y_mid, z_mid, t_i)

Run directly:
    python tests/test_quasi_static.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.gcode.parser import GCodeParser
from src.gcode.geometry import GeometryBuilder
from solver.thermal_solver import ThermalSolver

TESTS_DIR  = Path(__file__).parent
GCODE_FILE = TESTS_DIR / "20mmbox.gcode"
N_STEPS    = 10
LAYER_HEIGHT   = 0.2    # mm
NOZZLE_DIA     = 0.4    # mm
N_WALKS        = 256


def midpoint(move: dict) -> np.ndarray:
    return np.array([
        (move['x1'] + move['x2']) / 2.0,
        (move['y1'] + move['y2']) / 2.0,
        move['z'] - LAYER_HEIGHT / 2.0,   # vertical centre of bead
    ], dtype=np.float32)


def run_quasi_static():
    moves  = GCodeParser().parse(str(GCODE_FILE))
    gb     = GeometryBuilder(nozzle_diameter=NOZZLE_DIA, layer_height=LAYER_HEIGHT)
    solver = ThermalSolver(T_bed=60.0, T_nozzle=200.0, T_ambient=20.0,
                           n_walks=N_WALKS)

    results = []   # list of (t, x, y, z, T)

    t = 0
    for move in moves:
        prev_count = gb._vertex_count
        gb.add_move(move)

        # skip if GeometryBuilder rejected the move (zero-length bead)
        if gb._vertex_count == prev_count:
            continue

        verts, faces = gb.get_mesh()
        qpt = midpoint(move)

        # accumulated mesh is non-watertight once more than one bead exists
        T = solver.solve(verts, faces, qpt, watertight=(t == 0))

        results.append({
            't': t,
            'x': float(qpt[0]), 'y': float(qpt[1]), 'z': float(qpt[2]),
            'T': T,
            'dt_s': np.hypot(move['x2'] - move['x1'],
                             move['y2'] - move['y1']) / (move['f'] / 60.0),
        })
        print(f"  t={t:2d}  ({qpt[0]:6.2f}, {qpt[1]:6.2f}, {qpt[2]:.3f}) mm"
              f"  T = {T:6.1f} °C")
        t += 1
        if t >= N_STEPS:
            break

    return results


def plot_results(results):
    ts  = [r['t'] for r in results]
    Ts  = [r['T'] for r in results]
    xs  = [r['x'] for r in results]
    ys  = [r['y'] for r in results]
    zs  = [r['z'] for r in results]
    dts = np.cumsum([r['dt_s'] for r in results])

    colormap = cm.plasma
    norm     = mcolors.Normalize(vmin=min(Ts), vmax=max(Ts))

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ---- Plot 1: T vs time step index ---------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ts, Ts, 'o-', color='tomato', lw=2, ms=8)
    for t, T in zip(ts, Ts):
        ax1.annotate(f'{T:.1f}°C', (t, T), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=7)
    ax1.set_xlabel('Time step (move index)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('T at bead midpoint vs time step')
    ax1.set_xticks(ts)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ---- Plot 2: T vs cumulative real time ----------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(dts, Ts, 's-', color='steelblue', lw=2, ms=8)
    ax2.set_xlabel('Cumulative time (s)')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('T at bead midpoint vs real time')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # ---- Plot 3: 3D scatter — position coloured by T ------------------
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    sc = ax3.scatter(xs, ys, zs,
                     c=Ts, cmap=colormap, norm=norm, s=80, zorder=5)
    for i, r in enumerate(results):
        ax3.text(r['x'], r['y'], r['z'], f" t={r['t']}", fontsize=7)
    fig.colorbar(sc, ax=ax3, shrink=0.6, label='T (°C)')
    ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Y (mm)'); ax3.set_zlabel('Z (mm)')
    ax3.set_title('Query point positions coloured by T')

    # ---- Plot 4: T vs X position (shows spatial gradient) -------------
    ax4 = fig.add_subplot(gs[1, 1])
    sc4 = ax4.scatter(xs, Ts, c=ts, cmap='viridis', s=80, zorder=5)
    ax4.plot(xs, Ts, '--', color='gray', lw=1, alpha=0.5)
    fig.colorbar(sc4, ax=ax4, label='Time step')
    ax4.set_xlabel('X position of bead midpoint (mm)')
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title('T vs X position  (colour = time step)')
    ax4.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(
        f'Quasi-static thermal simulation — first {N_STEPS} time steps\n'
        f'(steady-state Laplace on accumulated bead geometry, WoS/zombie)',
        fontsize=12)
    plt.show()


if __name__ == "__main__":
    print(f"Running quasi-static simulation for {N_STEPS} time steps...")
    results = run_quasi_static()
    print("\nSummary:")
    print(f"  T range: {min(r['T'] for r in results):.1f} — "
          f"{max(r['T'] for r in results):.1f} °C")
    print(f"  Total simulated time: {sum(r['dt_s'] for r in results):.3f} s")
    plot_results(results)