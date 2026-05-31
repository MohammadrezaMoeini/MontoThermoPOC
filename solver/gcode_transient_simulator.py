"""
GCodeTransientSimulator: drives TransientThermalSolver bead-by-bead from
parsed GCode moves.

For each extrusion move the simulator:
  1. Builds the single-bead cuboid mesh.
  2. Computes the deposition duration from feed rate and bead length.
  3. Runs the transient solver during deposition (nozzle on top → T_nozzle BC).
  4. Runs a fixed number of cooling steps (nozzle gone → convection BC).
  5. Passes the mean final temperature forward as the initial condition for
     the next bead (approximates pre-heating from neighbouring material).
"""

import numpy as np
from .transient_thermal_solver import TransientThermalSolver
from src.gcode.geometry import GeometryBuilder


class GCodeTransientSimulator:

    def __init__(self,
                 solver:           TransientThermalSolver,
                 nozzle_diameter:  float = 0.4,   # mm
                 layer_height:     float = 0.2,   # mm
                 n_cooling_steps:  int   = 5):
        self.solver          = solver
        self.nozzle_diameter = nozzle_diameter
        self.layer_height    = layer_height
        self.n_cooling_steps = n_cooling_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, moves: list[dict], max_beads: int = None) -> list[dict]:
        """
        Simulate up to *max_beads* extrusion moves.

        Returns a list of per-bead result dicts:
            bead_idx        : int   — index of this bead in the move list
            move            : dict  — original GCode move
            dt_deposition   : float — time to deposit this bead [s]
            t_start         : float — global time at start of deposition [s]
            t_end           : float — global time after cooling [s]
            T_mid_history   : list  — T at bead midpoint at each solver step [°C]
            T_mid_times     : list  — global times for T_mid_history [s]
            T_final_mean    : float — mean temperature after cooling [°C]
            points          : (n_pts,3) array — interior query points [mm]
            T_deposition    : (n_dep+1, n_pts) array [°C]
            T_cooling       : (n_cool+1, n_pts) array [°C]
        """
        results   = []
        t_global  = 0.0
        T_ic      = self.solver.T_ambient  # scalar IC carried between beads

        for bead_idx, move in enumerate(moves):
            if max_beads is not None and bead_idx >= max_beads:
                break

            verts, faces = self._single_bead_mesh(move)
            if verts is None:
                continue

            # Time to deposit this bead [s]
            length = np.hypot(move['x2'] - move['x1'], move['y2'] - move['y1'])
            dt_dep = length / (move['f'] / 60.0)

            # Number of solver steps for deposition phase
            n_dep = max(1, round(dt_dep / self.solver.dt))

            t_start = t_global

            # ---- Deposition phase: nozzle on top -------------------------
            res_dep = self.solver.solve_transient(
                verts, faces,
                n_steps=n_dep,
                is_current_bead_schedule=[True] * n_dep,
                T_initial=T_ic)

            t_global += n_dep * self.solver.dt

            # ---- Cooling phase: nozzle has moved away --------------------
            T_after_dep = res_dep['T'][-1]   # (n_pts,) array
            res_cool = self.solver.solve_transient(
                verts, faces,
                n_steps=self.n_cooling_steps,
                is_current_bead_schedule=[False] * self.n_cooling_steps,
                T_initial=T_after_dep)

            t_global += self.n_cooling_steps * self.solver.dt
            t_end = t_global

            # ---- Midpoint temperature history ----------------------------
            mid_idx = self._nearest_midpoint_idx(res_dep['points'], move)

            dep_times  = t_start + res_dep['times']
            cool_times = (t_start + n_dep * self.solver.dt) + res_cool['times']

            T_mid_dep  = res_dep['T'][:, mid_idx]
            T_mid_cool = res_cool['T'][:, mid_idx]

            # Merge (skip duplicate boundary between phases)
            T_mid_times   = np.concatenate([dep_times,   cool_times[1:]])
            T_mid_history = np.concatenate([T_mid_dep,   T_mid_cool[1:]])

            # Carry forward mean final T as IC for next bead
            T_ic = float(res_cool['T'][-1].mean())

            results.append({
                'bead_idx':      bead_idx,
                'move':          move,
                'dt_deposition': dt_dep,
                't_start':       t_start,
                't_end':         t_end,
                'T_mid_history': T_mid_history,
                'T_mid_times':   T_mid_times,
                'T_final_mean':  T_ic,
                'points':        res_dep['points'],
                'T_deposition':  res_dep['T'],
                'T_cooling':     res_cool['T'],
            })

            print(f"  bead {bead_idx:3d}  layer={move['layer']}  "
                  f"dt_dep={dt_dep:.2f}s  "
                  f"T_mid: {T_mid_dep[0]:.1f}→{T_mid_dep[-1]:.1f}°C  "
                  f"(cool→{T_mid_cool[-1]:.1f}°C)")

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_bead_mesh(self, move: dict):
        """Build the cuboid mesh for a single GCode move."""
        gb = GeometryBuilder(
            nozzle_diameter=self.nozzle_diameter,
            layer_height=self.layer_height)
        gb.add_move(move)
        verts, faces = gb.get_mesh()
        if len(verts) == 0:
            return None, None
        return verts.astype(np.float32), faces.astype(np.int32)

    def _nearest_midpoint_idx(self, points: np.ndarray, move: dict) -> int:
        """Return the index of the grid point closest to the bead midpoint."""
        mid = np.array([
            (move['x1'] + move['x2']) / 2.0,
            (move['y1'] + move['y2']) / 2.0,
            move['z'] - self.layer_height / 2.0,
        ], dtype=np.float32)
        dists = np.linalg.norm(points - mid, axis=1)
        return int(np.argmin(dists))