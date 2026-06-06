"""
TransientThermalSolver: backward-Euler time integration of the heat equation
using zombie Walk-on-Stars.

At each step solves the screened-Poisson (Yukawa) equation:
    ∇²T^(n+1) − σ · T^(n+1) = −σ · T^n
where σ = 1/(α · Δt),  α = k/(ρ·c_p)  in mm²/s.

This is the backward-Euler discretisation of  ∂T/∂t = α · ∇²T,
which is unconditionally stable for any Δt > 0.
"""

import numpy as np
import zombie
from .thermal_solver import ThermalSolver

_DIM      = 3
_CHANNELS = 1


class TransientThermalSolver(ThermalSolver):

    def __init__(self,
                 T_bed:      float = 60.0,
                 T_nozzle:   float = 200.0,
                 T_ambient:  float = 20.0,
                 h:          float = 25.0,    # convection coeff  W/(m²·K)
                 k_cond:     float = 0.2,     # PLA conductivity  W/(m·K)
                 n_walks:    int   = 128,
                 dt:         float = 1.0,     # time step  [s]
                 rho:        float = 1240.0,  # PLA density  [kg/m³]
                 cp:         float = 1800.0,  # PLA specific heat  [J/(kg·K)]
                 grid_shape: tuple = (8, 4, 4)):
        super().__init__(T_bed, T_nozzle, T_ambient, h, k_cond, n_walks)
        self.dt    = dt
        # α = k/(ρ·cp) in m²/s → mm²/s (×1e6)
        self.alpha = (k_cond / (rho * cp)) * 1e6
        self.sigma = 1.0 / (self.alpha * dt)  # screened-Poisson coefficient [mm⁻²]
        self.grid_shape = grid_shape           # (nx, ny, nz) interior points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_transient(self,
                        vertices:  np.ndarray,
                        faces:     np.ndarray,
                        n_steps:   int,
                        is_current_bead_schedule=None,
                        T_initial=None,
                        query_pts=None) -> dict:
        """
        Run n_steps of backward-Euler time integration on a bead mesh.

        is_current_bead_schedule : list/array of bools, length n_steps.
            True  → nozzle on top face (Dirichlet T_nozzle).
            False → top face cools by convection.
            Defaults to all True.

        T_initial : scalar or (nx*ny*nz,) array [°C].
            Initial temperature field on the interior grid. Scalar applies
            uniformly. Defaults to T_ambient.

        query_pts : (n_pts, 3) array [mm], optional.
            Exact interior coordinates to evaluate. When provided, T is
            returned at these points; the interior grid is still maintained
            internally to supply the screened-Poisson source term.
            Points must lie strictly inside the mesh.

        Returns:
            times  : (n_steps+1,) array [s]
            T      : (n_steps+1, n_pts) array [°C]
            points : (n_pts, 3) array of interior query coordinates [mm]
        """
        verts = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces,    dtype=np.int32)

        z_bot = float(verts[:, 2].min())
        z_top = float(verts[:, 2].max())
        eps   = (z_top - z_bot) * 0.1

        # The interior grid is always built: it tracks T_grid used as the
        # source buffer (σ·T^n term) in the screened-Poisson equation.
        grid_pts, gx, gy, gz = self._build_interior_grid(verts, z_bot, z_top)
        n_grid = len(grid_pts)

        if query_pts is None:
            # No custom points: solve only at grid points (existing behaviour).
            combined_pts = grid_pts
            n_extra      = 0
            output_pts   = grid_pts
        else:
            query_pts  = np.asarray(query_pts, dtype=np.float32)
            n_extra    = len(query_pts)
            # Solve at grid points AND custom query points in one WoS call.
            combined_pts = np.vstack([grid_pts, query_pts])
            output_pts   = query_pts

        n_output = len(output_pts)

        # T_grid tracks the temperature at the interior grid (for source buffer).
        if T_initial is None:
            T_grid = np.full(n_grid, self.T_ambient, dtype=np.float64)
        elif np.isscalar(T_initial):
            T_grid = np.full(n_grid, float(T_initial), dtype=np.float64)
        else:
            T_arr = np.asarray(T_initial, dtype=np.float64)
            if T_arr.shape == (n_grid,):
                T_grid = T_arr
            else:
                raise ValueError(
                    f"T_initial shape {T_arr.shape} does not match grid size ({n_grid},)"
                )

        # Initial snapshot at output points (uniform at T_initial mean).
        T_output = np.full(n_output, float(T_grid.mean()), dtype=np.float64)

        times     = [0.0]
        T_history = [T_output.copy()]

        for step in range(n_steps):
            is_current = (is_current_bead_schedule[step]
                          if is_current_bead_schedule is not None else True)
            T_all = self._time_step(verts, faces, combined_pts, gx, gy, gz,
                                    T_grid, z_bot, z_top, eps, bool(is_current))
            T_grid   = T_all[:n_grid]
            T_output = T_all[n_grid:] if n_extra > 0 else T_all
            times.append((step + 1) * self.dt)
            T_history.append(T_output.copy())

        return {
            'times':  np.array(times),
            'T':      np.array(T_history),   # shape (n_steps+1, n_output)
            'points': output_pts,            # shape (n_output, 3)
        }

    def solve_incremental_deposition(self,
                                     vertices:     np.ndarray,
                                     faces:        np.ndarray,
                                     n_steps:      int,
                                     bead_y1:      float,
                                     bead_y2:      float,
                                     T_substrate:  float = None,
                                     query_pts:    np.ndarray = None) -> dict:
        """
        Moving-nozzle deposition along the y-axis.

        The nozzle travels from bead_y1 to bead_y2 in n_steps time steps.
        At each step k the nozzle tip is at  y_tip = bead_y1 + (k+1)*dy.

        Interior grid points that the nozzle has NOT yet passed are reset to
        T_nozzle in the source buffer (freshly deposited material).  Points
        already behind the nozzle carry their temperature from the previous
        step and begin cooling naturally.

        Returns
        -------
        times        : (n_steps+1,) [s]
        T            : (n_steps+1, n_output) temperatures at output points
        points       : (n_output, 3)
        T_grid_final : (n_grid,) final interior-grid T (pass to solve_transient
                       as T_initial for the subsequent cooling phase)
        nozzle_step  : (n_output,) index k after which point j has valid
                       (freshly deposited) data; use T[nozzle_step[j]+1:, j]
        """
        verts = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces,    dtype=np.int32)

        z_bot = float(verts[:, 2].min())
        z_top = float(verts[:, 2].max())
        eps   = (z_top - z_bot) * 0.1

        grid_pts, gx, gy, gz = self._build_interior_grid(verts, z_bot, z_top)
        n_grid = len(grid_pts)
        grid_y = grid_pts[:, 1]

        if query_pts is not None:
            query_pts  = np.asarray(query_pts, dtype=np.float32)
            n_extra    = len(query_pts)
            combined   = np.vstack([grid_pts, query_pts])
            output_pts = query_pts
        else:
            n_extra    = 0
            combined   = grid_pts
            output_pts = grid_pts
        n_output = len(output_pts)
        output_y = output_pts[:, 1]

        T_sub  = float(self.T_ambient if T_substrate is None else T_substrate)
        T_grid = np.full(n_grid,  T_sub, dtype=np.float64)

        # Step when the nozzle first passes each output point (0-indexed).
        # After this step the point has valid (freshly deposited) data.
        bead_len = bead_y2 - bead_y1        # signed
        nozzle_step = np.zeros(n_output, dtype=int)
        for j in range(n_output):
            frac = np.clip((output_y[j] - bead_y1) / bead_len, 0.0, 1.0)
            nozzle_step[j] = max(0, int(np.ceil(frac * n_steps)) - 1)

        times     = [0.0]
        T_history = [np.full(n_output, T_sub, dtype=np.float64)]

        dy = bead_len / n_steps   # signed step size along y

        for k in range(n_steps):
            y_tip_prev = bead_y1 + k * dy

            # Grid points not yet deposited → reset to T_nozzle
            not_yet = grid_y > y_tip_prev if dy >= 0 else grid_y < y_tip_prev
            T_grid  = np.where(not_yet, self.T_nozzle, T_grid)

            T_all    = self._time_step(verts, faces, combined, gx, gy, gz,
                                       T_grid, z_bot, z_top, eps, True)
            T_grid   = T_all[:n_grid]
            T_output = T_all[n_grid:] if n_extra > 0 else T_all

            times.append((k + 1) * self.dt)
            T_history.append(T_output.copy())

        return {
            'times':        np.array(times),
            'T':            np.array(T_history),   # (n_steps+1, n_output)
            'points':       output_pts,
            'T_grid_final': T_grid.copy(),
            'nozzle_step':  nozzle_step,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_interior_grid(self, verts, z_bot, z_top):
        """Return (pts, xs, ys, zs): interior grid excluding boundary faces."""
        nx, ny, nz = self.grid_shape
        x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
        y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())

        xs = np.linspace(x_min, x_max, nx + 2)[1:-1]
        ys = np.linspace(y_min, y_max, ny + 2)[1:-1]
        zs = np.linspace(z_bot,  z_top,  nz + 2)[1:-1]

        pts = np.array(
            [[xs[xi], ys[yi], zs[zi]]
             for xi in range(nx)
             for yi in range(ny)
             for zi in range(nz)],
            dtype=np.float32)
        return pts, xs, ys, zs

    def _build_source_buffer(self, verts, T_prev, z_bot, z_top, is_current_bead):
        """
        Dense-grid source  f = σ · T^n  over the full bounding box.

        Grid includes one boundary layer on each face so that walks that
        terminate near a boundary read a reasonable T^n value.

        Indexing: buf[xi * my * mz + yi * mz + zi]  (C-order, x outer).
        """
        nx, ny, nz = self.grid_shape
        mx, my, mz = nx + 2, ny + 2, nz + 2

        x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
        y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())

        u_full = np.full((mx, my, mz), self.T_ambient, dtype=np.float32)

        # Absorbing BCs on boundary faces
        u_full[:, :, 0]      = self.T_bed
        u_full[:, :, mz - 1] = self.T_nozzle if is_current_bead else self.T_ambient

        # Interior: T^n from previous step
        u_full[1:mx-1, 1:my-1, 1:mz-1] = T_prev.reshape(nx, ny, nz).astype(np.float32)

        source_buf = (self.sigma * u_full).ravel().astype(np.float32)
        shape = np.array([mx, my, mz], dtype=np.int32)
        d_min = np.array([x_min, y_min, z_bot], dtype=np.float32)
        d_max = np.array([x_max, y_max, z_top], dtype=np.float32)
        return source_buf, shape, d_min, d_max

    def _build_transient_pde(self, verts, faces, T_prev, z_bot, z_top,
                              eps, is_current_bead):
        """PDE with absorption coefficient (screened-Poisson) + T^n source."""
        source_buf, source_shape, d_min, d_max = \
            self._build_source_buffer(verts, T_prev, z_bot, z_top, is_current_bead)

        absorbing_pos, absorbing_idx, reflecting_pos, reflecting_idx = \
            self._partition_mesh(verts, faces, z_bot, z_top, eps, is_current_bead)

        # Dirichlet grid (same 3-level Z structure as steady-state solver)
        if is_current_bead:
            dir_buf = np.array([self.T_bed, self.T_ambient, self.T_nozzle],
                                dtype=np.float32)
        else:
            dir_buf = np.array([self.T_bed, self.T_bed, self.T_bed],
                                dtype=np.float32)
        dir_shape = np.array([1, 1, 3], dtype=np.int32)
        dir_dmin  = np.array([verts[:, 0].min(), verts[:, 1].min(), z_bot],
                              dtype=np.float32)
        dir_dmax  = np.array([verts[:, 0].max(), verts[:, 1].max(), z_top],
                              dtype=np.float32)

        pde = zombie.Core.PDE(dim=_DIM, channels=_CHANNELS)
        pde.absorption_coeff = self.sigma
        pde.source = zombie.Utils.get_dense_grid_source_callback(
            source_buf, source_shape, d_min, d_max,
            dim=_DIM, channels=_CHANNELS)
        pde.dirichlet = zombie.Utils.get_dense_grid_dirichlet_callback(
            dir_buf, dir_shape, dir_dmin, dir_dmax,
            dim=_DIM, channels=_CHANNELS)
        pde.robin = zombie.Core.get_constant_robin_callback(
            self.kappa, dim=_DIM, channels=_CHANNELS)
        pde.robin_coeff = zombie.Core.get_constant_robin_coefficient_callback(
            self.mu, dim=_DIM)
        pde.has_reflecting_boundary_conditions = \
            zombie.Core.get_constant_indicator_callback(True, dim=_DIM)
        pde.are_robin_conditions_pure_neumann  = False
        pde.are_robin_coeffs_nonnegative       = (self.mu >= 0)

        refs = (absorbing_pos, absorbing_idx, reflecting_pos, reflecting_idx,
                source_buf, dir_buf, dir_shape, dir_dmin, dir_dmax)
        return pde, refs

    def _time_step(self, verts, faces, query_pts, gx, gy, gz,
                   T_prev, z_bot, z_top, eps, is_current_bead) -> np.ndarray:
        """One backward-Euler step: solve screened-Poisson for all grid points."""
        pde, refs = self._build_transient_pde(
            verts, faces, T_prev, z_bot, z_top, eps, is_current_bead)
        absorbing_pos, absorbing_idx, reflecting_pos, reflecting_idx = refs[:4]

        bbox = zombie.Utils.compute_bounding_box(
            zombie.Float3List(verts), False, 1.0, dim=_DIM)
        gq, gq_refs = self._build_geometric_queries(
            bbox, absorbing_pos, absorbing_idx, reflecting_pos, reflecting_idx)

        has_reflecting = len(reflecting_idx) > 0
        walk_settings  = zombie.Solvers.WalkSettings(
            1e-3, 1e-3, 1e-3,
            0.0, np.inf,
            1024, 0, 1024,
            False,
            True, True, False,
            False,
            not has_reflecting,
            False, False)

        n_pts = len(query_pts)
        sp_list = []
        ss_list = []
        for qpt in query_pts:
            qpt_f = np.asarray(qpt, dtype=np.float32)
            d_abs = gq.compute_dist_to_absorbing_boundary(qpt_f, False)
            d_ref = gq.compute_dist_to_reflecting_boundary(qpt_f, False)
            sp = zombie.Solvers.SamplePoint(
                qpt_f, np.zeros(_DIM, dtype=np.float32),
                zombie.Solvers.SampleType.InDomain,
                zombie.Solvers.EstimationQuantity.Solution,
                1.0, d_abs, d_ref,
                dim=_DIM, channels=_CHANNELS)
            ss = zombie.Solvers.SampleStatistics(dim=_DIM, channels=_CHANNELS)
            sp_list.append(sp)
            ss_list.append(ss)

        sample_pts   = zombie.Solvers.SamplePointList(sp_list, dim=_DIM, channels=_CHANNELS)
        sample_stats = zombie.Solvers.SampleStatisticsList(ss_list, dim=_DIM, channels=_CHANNELS)
        n_walks_list = zombie.IntList([self.n_walks] * n_pts)

        pb     = zombie.Utils.ProgressBar(n_pts)
        rp     = zombie.Utils.get_report_progress_callback(pb)
        solver = zombie.Solvers.WalkOnStars(gq, dim=_DIM, channels=_CHANNELS)
        solver.solve(pde, walk_settings, n_walks_list, sample_pts, sample_stats,
                     False, rp)
        pb.finish()

        return np.array([sample_stats[i].get_estimated_solution() for i in range(n_pts)])