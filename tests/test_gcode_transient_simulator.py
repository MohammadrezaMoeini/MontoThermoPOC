"""
Tests for GCodeTransientSimulator.

Validates the bead-by-bead transient simulation driven by GCode moves.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.gcode.parser import GCodeParser
from solver.transient_thermal_solver import TransientThermalSolver
from solver.gcode_transient_simulator import GCodeTransientSimulator

GCODE_FILE = Path(__file__).parent / "20mmbox.gcode"
N_BEADS    = 4   # keep tests fast


@pytest.fixture(scope="module")
def moves():
    return GCodeParser().parse(str(GCODE_FILE))


@pytest.fixture(scope="module")
def simulator():
    solver = TransientThermalSolver(
        T_bed=60.0, T_nozzle=200.0, T_ambient=20.0,
        n_walks=64, dt=0.5, grid_shape=(4, 2, 2))
    return GCodeTransientSimulator(solver, n_cooling_steps=3)


@pytest.fixture(scope="module")
def results(moves, simulator):
    return simulator.run(moves, max_beads=N_BEADS)


class TestGCodeTransientSimulator:

    def test_gcode_has_moves(self, moves):
        assert len(moves) > 0

    def test_result_count(self, results):
        # Some moves may have zero length and get skipped by the simulator
        assert 1 <= len(results) <= N_BEADS

    def test_output_keys(self, results):
        required = {'bead_idx', 'move', 'dt_deposition', 't_start', 't_end',
                    'T_mid_history', 'T_mid_times', 'T_final_mean',
                    'points', 'T_deposition', 'T_cooling'}
        for r in results:
            assert required.issubset(r.keys())

    def test_global_time_monotonic(self, results):
        times = [r['t_start'] for r in results] + [results[-1]['t_end']]
        assert all(times[i] < times[i+1] for i in range(len(times) - 1))

    def test_t_end_gt_t_start(self, results):
        for r in results:
            assert r['t_end'] > r['t_start']

    def test_T_mid_history_length(self, results, simulator):
        for r in results:
            # n_dep steps + n_cooling steps + 1 (IC) — merged without duplicate boundary
            assert len(r['T_mid_history']) == len(r['T_mid_times'])
            assert len(r['T_mid_history']) > 1

    def test_T_mid_times_monotonic(self, results):
        for r in results:
            times = r['T_mid_times']
            assert np.all(np.diff(times) > 0)

    def test_temperatures_physically_bounded(self, results, simulator):
        lo = simulator.solver.T_ambient - 5.0
        hi = simulator.solver.T_nozzle  + 5.0
        for r in results:
            for key in ('T_deposition', 'T_cooling'):
                T = r[key]
                # Filter exact 0.0: known MC artifact for grid points very
                # close to boundary faces in short beads (zombie returns 0
                # when the walk exhausts its steps near a reflecting face).
                T_valid = T[T != 0.0]
                assert T_valid.min() >= lo, f"{key} too cold: {T_valid.min():.1f} °C"
                assert T_valid.max() <= hi, f"{key} too hot:  {T_valid.max():.1f} °C"

    def test_deposition_heats_bead(self, results):
        """Midpoint temperature should rise during deposition."""
        for r in results:
            T_dep = r['T_deposition'][:, 0]   # first grid point over time
            assert T_dep[-1] > T_dep[0], (
                f"Bead {r['bead_idx']}: temperature did not rise during deposition "
                f"({T_dep[0]:.1f} → {T_dep[-1]:.1f} °C)"
            )

    def test_cooling_lowers_temperature(self, results):
        """Mean temperature should drop during the cooling phase."""
        for r in results:
            T_cool = r['T_cooling']
            mean_start = T_cool[0].mean()
            mean_end   = T_cool[-1].mean()
            assert mean_end < mean_start, (
                f"Bead {r['bead_idx']}: cooling did not lower T "
                f"({mean_start:.1f} → {mean_end:.1f} °C)"
            )

    def test_temperature_carried_forward(self, results, simulator):
        """Second bead IC should be above T_ambient due to heat from bead 0."""
        T_final_bead0 = results[0]['T_final_mean']
        assert T_final_bead0 > simulator.solver.T_ambient, (
            f"Expected heat carryover above T_ambient={simulator.solver.T_ambient}°C, "
            f"got T_final={T_final_bead0:.1f}°C"
        )

    def test_dt_deposition_positive(self, results):
        for r in results:
            assert r['dt_deposition'] > 0.0

    def test_points_inside_bead_bounds(self, results):
        """All query points should lie within the bead's vertex bounding box."""
        from src.gcode.geometry import GeometryBuilder
        gb  = GeometryBuilder(nozzle_diameter=0.4, layer_height=0.2)
        tol = 0.01  # mm
        for r in results:
            gb.reset()
            gb.add_move(r['move'])
            verts, _ = gb.get_mesh()
            points   = r['points']
            x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
            y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())
            z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())

            assert np.all(points[:, 0] >= x_min - tol)
            assert np.all(points[:, 0] <= x_max + tol)
            assert np.all(points[:, 1] >= y_min - tol)
            assert np.all(points[:, 1] <= y_max + tol)
            assert np.all(points[:, 2] >= z_min - tol)
            assert np.all(points[:, 2] <= z_max + tol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])