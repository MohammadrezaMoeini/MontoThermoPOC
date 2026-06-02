"""
Example 06: Temperature vs time at three vertical positions inside a bead.

The three points all share (x_mid, y_mid) and differ only in z:
  - z_bottom : nearest interior point to the bed  (z ≈ z_bot + dz)
  - z_middle : centre of the bead cross-section   (z ≈ (z_bot + z_top) / 2)
  - z_top    : nearest interior point to the nozzle (z ≈ z_top - dz)

This shows how the vertical thermal gradient evolves over time:
  - z_top heats up fast during deposition (Dirichlet T_nozzle on the top face)
    then drops sharply when the nozzle leaves.
  - z_bottom stays close to T_bed (Dirichlet on the bottom face).
  - z_middle shows the intermediate transient behaviour.

Run with:
    source ~/.MontoThermoPOC312/bin/activate
    python experimentDev/example06/example06.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.gcode.parser import GCodeParser
from solver.transient_thermal_solver import TransientThermalSolver
from solver.gcode_transient_simulator import GCodeTransientSimulator

# ── Parameters ────────────────────────────────────────────────────────────────
GCODE_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', '20mmbox.gcode')
MAX_BEADS  = 30      # scan this many beads to find a long one
N_WALKS    = 256
DT         = 0.2     # time step [s]
N_COOLING  = 15      # cooling steps after deposition
OUT_DIR    = os.path.dirname(__file__)

# 1 x-point (x_mid), 1 y-point (y_mid), 3 z-points (bottom / middle / top)
GRID_SHAPE = (1, 1, 3)

# ── Run ───────────────────────────────────────────────────────────────────────
moves = GCodeParser().parse(GCODE_FILE)
print(f"Parsed {len(moves)} moves. Simulating up to {MAX_BEADS} beads...")

solver = TransientThermalSolver(
    T_bed=60.0, T_nozzle=200.0, T_ambient=20.0,
    h=25.0, k_cond=0.2,
    n_walks=N_WALKS, dt=DT,
    rho=1240.0, cp=1800.0,
    grid_shape=GRID_SHAPE)

sim = GCodeTransientSimulator(
    solver, nozzle_diameter=0.4, layer_height=0.2,
    n_cooling_steps=N_COOLING)

results = sim.run(moves, max_beads=MAX_BEADS)

# Pick the bead with the most deposition steps (longest bead)
r = max(results, key=lambda x: x['T_deposition'].shape[0])
n_dep = r['T_deposition'].shape[0]   # includes IC row (step 0)

print(f"\nSelected bead {r['bead_idx']}  "
      f"(deposition time ≈ {r['dt_deposition']:.2f}s, "
      f"{n_dep - 1} deposition steps + {N_COOLING} cooling steps)")

# ── Extract the 3 z-point indices ─────────────────────────────────────────────
# With grid_shape=(1,1,3): point index = xi*ny*nz + yi*nz + zi = zi
# zi=0 → z_bottom, zi=1 → z_middle, zi=2 → z_top
points = r['points']   # shape (3, 3) — 3 grid points, each (x,y,z)

z_bottom_val = float(points[0, 2])
z_middle_val = float(points[1, 2])
z_top_val    = float(points[2, 2])

print(f"Grid z-coordinates: "
      f"bottom={z_bottom_val:.3f}mm, "
      f"middle={z_middle_val:.3f}mm, "
      f"top={z_top_val:.3f}mm")

# ── Build full T history (deposition + cooling) ───────────────────────────────
T_dep  = r['T_deposition']          # (n_dep,   3)
T_cool = r['T_cooling']             # (n_cool+1, 3)
T_full = np.vstack([T_dep, T_cool[1:]])   # (n_dep + n_cool, 3)

n_total = T_full.shape[0]
times   = np.arange(n_total) * DT   # time relative to deposition start [s]

# T for each of the 3 z-points
T_bottom = T_full[:, 0]
T_middle = T_full[:, 1]
T_top    = T_full[:, 2]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

t_dep_end = (n_dep - 1) * DT   # time when nozzle leaves

# Shade deposition vs cooling regions
ax.axvspan(0, t_dep_end,
           alpha=0.08, color='tomato',    label='Deposition phase')
ax.axvspan(t_dep_end, times[-1],
           alpha=0.08, color='steelblue', label='Cooling phase')
ax.axvline(t_dep_end, color='black', lw=1.2, ls='--', alpha=0.6)
ax.text(t_dep_end + 0.05, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 25,
        'nozzle\nleaves', fontsize=8, va='bottom', color='black', alpha=0.7)

# Three temperature curves
ax.plot(times, T_bottom, color='steelblue', lw=2.5, marker='o', ms=4,
        label=f'z_bottom = {z_bottom_val:.3f} mm  (near bed)')
ax.plot(times, T_middle, color='seagreen',  lw=2.5, marker='s', ms=4,
        label=f'z_middle = {z_middle_val:.3f} mm')
ax.plot(times, T_top,    color='tomato',    lw=2.5, marker='^', ms=4,
        label=f'z_top    = {z_top_val:.3f} mm  (near nozzle)')

# Reference lines
ax.axhline(solver.T_nozzle,  color='tomato',    lw=1, ls=':',
           alpha=0.6, label=f'T_nozzle = {solver.T_nozzle:.0f}°C')
ax.axhline(solver.T_bed,     color='steelblue', lw=1, ls=':',
           alpha=0.6, label=f'T_bed = {solver.T_bed:.0f}°C')
ax.axhline(solver.T_ambient, color='gray',       lw=1, ls=':',
           alpha=0.6, label=f'T_ambient = {solver.T_ambient:.0f}°C')

ax.set_xlabel('Time [s]  (relative to start of bead deposition)', fontsize=13)
ax.set_ylabel('Temperature [°C]', fontsize=13)
ax.set_title(
    f'Temperature vs time at three vertical positions — bead {r["bead_idx"]}\n'
    f'x = x_mid, y = y_mid  |  {N_WALKS} walks, Δt = {DT}s',
    fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()

out_path = os.path.join(OUT_DIR, 'T_vs_time.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()