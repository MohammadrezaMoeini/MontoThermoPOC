"""
Tests for TransientThermalSolver.solve_incremental_deposition.

Covers output structure, nozzle_step correctness, temperature physics,
and the no-query-pts fallback.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from solver.transient_thermal_solver import TransientThermalSolver


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_bead_mesh(y_len=10.0, x_half=0.2, z_bot=0.0, z_top=0.4):
    """Closed triangular mesh for a bead-like cuboid along the y-axis."""
    verts = np.array([
        [-x_half, 0.0,   z_bot], [x_half, 0.0,   z_bot],
        [ x_half, y_len, z_bot], [-x_half, y_len, z_bot],
        [-x_half, 0.0,   z_top], [x_half, 0.0,   z_top],
        [ x_half, y_len, z_top], [-x_half, y_len, z_top],
    ], dtype=np.float32)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],   # bottom
        [4, 5, 6], [4, 6, 7],   # top
        [0, 1, 5], [0, 5, 4],   # front  (y=0)
        [2, 3, 7], [2, 7, 6],   # back   (y=y_len)
        [0, 4, 7], [0, 7, 3],   # left
        [1, 2, 6], [1, 6, 5],   # right
    ], dtype=np.int32)
    return verts, faces


@pytest.fixture(scope='module')
def solver():
    return TransientThermalSolver(
        T_bed=52.0, T_nozzle=210.0, T_ambient=20.0,
        h=71.0, k_cond=0.11,
        rho=1250.0, cp=1590.0,
        n_walks=32, dt=0.5,
        grid_shape=(4, 2, 2),
    )


@pytest.fixture(scope='module')
def bead_mesh():
    return make_bead_mesh(y_len=10.0)


@pytest.fixture(scope='module')
def three_query_pts():
    """One point near each of: bead start, mid, end."""
    return np.array([
        [0.0, 0.5, 0.2],   # near y=0 (start)
        [0.0, 5.0, 0.2],   # mid
        [0.0, 9.5, 0.2],   # near y=10 (end)
    ], dtype=np.float32)


@pytest.fixture(scope='module')
def result(solver, bead_mesh, three_query_pts):
    verts, faces = bead_mesh
    return solver.solve_incremental_deposition(
        verts, faces, n_steps=10,
        bead_y1=0.0, bead_y2=10.0,
        T_substrate=25.0,
        query_pts=three_query_pts,
    )


# ── Output structure ──────────────────────────────────────────────────────────

class TestOutputStructure:

    def test_required_keys(self, result):
        assert {'times', 'T', 'points', 'T_grid_final', 'nozzle_step'} <= set(result)

    def test_times_shape(self, result):
        assert result['times'].shape == (11,)   # n_steps + 1

    def test_times_starts_at_zero(self, result):
        assert result['times'][0] == pytest.approx(0.0)

    def test_times_monotonically_increasing(self, result):
        assert np.all(np.diff(result['times']) > 0)

    def test_times_step_equals_dt(self, result, solver):
        np.testing.assert_allclose(np.diff(result['times']), solver.dt, rtol=1e-6)

    def test_T_shape(self, result, three_query_pts):
        assert result['T'].shape == (11, len(three_query_pts))

    def test_nozzle_step_shape(self, result, three_query_pts):
        assert result['nozzle_step'].shape == (len(three_query_pts),)

    def test_T_grid_final_shape(self, solver, bead_mesh):
        verts, faces = bead_mesh
        res = solver.solve_incremental_deposition(
            verts, faces, n_steps=4, bead_y1=0.0, bead_y2=10.0)
        nx, ny, nz = solver.grid_shape
        assert res['T_grid_final'].shape == (nx * ny * nz,)

    def test_points_matches_query_pts(self, result, three_query_pts):
        np.testing.assert_array_equal(result['points'], three_query_pts)


# ── nozzle_step correctness ───────────────────────────────────────────────────

class TestNozzleStep:

    def test_start_point_has_lowest_nozzle_step(self, result):
        assert result['nozzle_step'][0] <= 1

    def test_end_point_has_highest_nozzle_step(self, result):
        assert result['nozzle_step'][2] >= 8

    def test_nozzle_steps_non_decreasing(self, result):
        ns = result['nozzle_step']
        assert ns[0] <= ns[1] <= ns[2]

    def test_nozzle_steps_within_valid_range(self, result):
        ns = result['nozzle_step']
        assert np.all(ns >= 0)
        assert np.all(ns < 10)   # n_steps=10

    def test_bead_start_point_gets_step_zero(self, solver, bead_mesh):
        verts, faces = bead_mesh
        qpt = np.array([[0.0, 0.0, 0.2]], dtype=np.float32)   # y=bead_y1
        res = solver.solve_incremental_deposition(
            verts, faces, n_steps=10, bead_y1=0.0, bead_y2=10.0,
            query_pts=qpt)
        assert res['nozzle_step'][0] == 0


# ── Temperature physics ───────────────────────────────────────────────────────

class TestTemperaturePhysics:

    def test_initial_row_equals_T_substrate(self, result):
        np.testing.assert_allclose(result['T'][0], 25.0, atol=1e-6)

    def test_all_T_within_physical_bounds(self, result, solver):
        T = result['T']
        assert T.min() >= solver.T_ambient - 10.0
        assert T.max() <= solver.T_nozzle  + 10.0

    def test_T_grid_final_within_physical_bounds(self, result, solver):
        T = result['T_grid_final']
        assert T.min() >= solver.T_ambient - 10.0
        assert T.max() <= solver.T_nozzle  + 10.0

    def test_high_T_right_after_nozzle_arrival(self, result, solver):
        for j in range(result['T'].shape[1]):
            k0 = int(result['nozzle_step'][j])
            T_first = result['T'][k0 + 1, j]
            assert T_first > solver.T_ambient + 30.0, (
                f"Point {j}: T just after nozzle = {T_first:.1f} °C, "
                f"expected > {solver.T_ambient + 30.0:.1f} °C")

    def test_mid_bead_point_cools_after_nozzle_passes(self, bead_mesh):
        # Use n_walks=64 and n_steps=14 to give the mid-bead point 7+ cooling steps.
        # Even with MC noise, the mean of the last half should be below the first.
        s = TransientThermalSolver(
            T_bed=52., T_nozzle=210., T_ambient=20.,
            h=71., k_cond=0.11, rho=1250., cp=1590.,
            n_walks=64, dt=0.5, grid_shape=(4, 2, 2))
        verts, faces = bead_mesh
        qpt = np.array([[0., 5., 0.2]], dtype=np.float32)
        res = s.solve_incremental_deposition(
            verts, faces, n_steps=14, bead_y1=0., bead_y2=10.,
            T_substrate=20., query_pts=qpt)
        k0      = int(res['nozzle_step'][0])
        T_trace = res['T'][k0 + 1:, 0]
        assert len(T_trace) >= 5, "Too few post-deposit steps"
        mean_first_two = T_trace[:2].mean()
        mean_last_two  = T_trace[-2:].mean()
        assert mean_last_two < mean_first_two, (
            f"Mid-bead point did not cool on average: "
            f"{mean_first_two:.1f} → {mean_last_two:.1f} °C")


# ── No query points — falls back to grid ─────────────────────────────────────

class TestNoQueryPoints:

    def test_T_shape_uses_grid_size(self, solver, bead_mesh):
        verts, faces = bead_mesh
        res = solver.solve_incremental_deposition(
            verts, faces, n_steps=4, bead_y1=0.0, bead_y2=10.0)
        nx, ny, nz = solver.grid_shape
        assert res['T'].shape == (5, nx * ny * nz)

    def test_T_grid_final_matches_last_T_row(self, solver, bead_mesh):
        verts, faces = bead_mesh
        res = solver.solve_incremental_deposition(
            verts, faces, n_steps=4, bead_y1=0.0, bead_y2=10.0)
        np.testing.assert_array_equal(res['T_grid_final'], res['T'][-1])

    def test_default_T_substrate_is_T_ambient(self, solver, bead_mesh):
        verts, faces = bead_mesh
        res = solver.solve_incremental_deposition(
            verts, faces, n_steps=2, bead_y1=0.0, bead_y2=10.0)
        np.testing.assert_allclose(res['T'][0], solver.T_ambient, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])