"""
Tests for TransientThermalSolver.

Validates that the backward-Euler WoS time-stepping:
  - produces temperatures within physically valid bounds
  - converges toward the steady-state solution as t → ∞
  - correctly handles the is_current_bead flag
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from solver.transient_thermal_solver import TransientThermalSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_box_mesh(x0, x1, y0, y1, z0, z1):
    """
    Closed triangular mesh for an axis-aligned box.
    Returns (vertices (8,3), faces (12,3)).
    """
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # top
    ], dtype=np.float32)

    faces = np.array([
        # bottom (z=z0)
        [0, 2, 1], [0, 3, 2],
        # top (z=z1)
        [4, 5, 6], [4, 6, 7],
        # front (y=y0)
        [0, 1, 5], [0, 5, 4],
        # back (y=y1)
        [2, 3, 7], [2, 7, 6],
        # left (x=x0)
        [0, 4, 7], [0, 7, 3],
        # right (x=x1)
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)

    return verts, faces


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransientThermalSolver:

    def setup_method(self):
        self.solver = TransientThermalSolver(
            T_bed=60.0,
            T_nozzle=200.0,
            T_ambient=20.0,
            h=25.0,
            k_cond=0.2,
            n_walks=64,
            dt=1.0,
            rho=1240.0,
            cp=1800.0,
            grid_shape=(4, 2, 2),
        )
        self.verts, self.faces = make_box_mesh(
            x0=0.0, x1=4.0,
            y0=0.0, y1=0.4,
            z0=0.0, z1=0.2,
        )

    def test_output_structure(self):
        result = self.solver.solve_transient(self.verts, self.faces, n_steps=2)

        assert 'times' in result
        assert 'T' in result
        assert 'points' in result

        assert result['times'].shape == (3,)   # t=0 + 2 steps
        assert result['T'].shape[0]   == 3
        assert result['points'].ndim  == 2
        assert result['points'].shape[1] == 3

    def test_temperatures_in_physical_bounds(self):
        result = self.solver.solve_transient(self.verts, self.faces, n_steps=3)
        T = result['T']

        T_lo = self.solver.T_ambient - 5.0   # small tolerance for MC noise
        T_hi = self.solver.T_nozzle  + 5.0
        assert T.min() >= T_lo, f"Temperature too low: {T.min():.2f} °C"
        assert T.max() <= T_hi, f"Temperature too high: {T.max():.2f} °C"

    def test_initial_condition_is_ambient(self):
        result = self.solver.solve_transient(self.verts, self.faces, n_steps=1)
        T0 = result['T'][0]
        np.testing.assert_allclose(T0, self.solver.T_ambient, atol=1e-6)

    def test_convergence_toward_steady_state(self):
        """Temperature should stabilise as steps increase (last few steps close)."""
        result = self.solver.solve_transient(self.verts, self.faces, n_steps=20)
        T = result['T']

        # With 64 MC walks, per-step noise ~10-20 °C; check trend over last 5 steps
        # rather than a tight single-step threshold.
        late_mean = T[-5:].mean(axis=1)  # mean T over grid for last 5 steps
        delta = np.abs(np.diff(late_mean)).max()
        assert delta < 30.0, (
            f"Solution has not converged: max |ΔT_mean| = {delta:.2f} °C in last 5 steps"
        )

    def test_current_vs_previous_bead(self):
        """Previous-bead BCs (top cools) should give lower mean T than current-bead."""
        result_current  = self.solver.solve_transient(
            self.verts, self.faces, n_steps=5,
            is_current_bead_schedule=[True] * 5)
        result_previous = self.solver.solve_transient(
            self.verts, self.faces, n_steps=5,
            is_current_bead_schedule=[False] * 5)

        T_current  = result_current['T'][-1].mean()
        T_previous = result_previous['T'][-1].mean()
        assert T_current > T_previous, (
            f"Current bead ({T_current:.1f} °C) should be hotter than "
            f"previous bead ({T_previous:.1f} °C)"
        )

    def test_sigma_computation(self):
        """σ = 1/(α·Δt) should be positive and finite."""
        assert self.solver.sigma > 0
        assert np.isfinite(self.solver.sigma)

    def test_alpha_units(self):
        """Thermal diffusivity for PLA should be in the range 0.05–0.15 mm²/s."""
        assert 0.05 < self.solver.alpha < 0.15, (
            f"Unexpected α = {self.solver.alpha:.4f} mm²/s for PLA"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])