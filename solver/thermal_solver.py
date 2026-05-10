"""
ThermalSolver: solves the steady-state heat equation on a single bead mesh
using the zombie Walk-on-Stars solver.

PDE: Laplace (Δu = 0)

Boundary conditions depend on whether the bead is currently under the nozzle
(current bead) or was deposited earlier and is cooling by convection.

Current bead (just deposited):
  - Bottom face (z ≈ z_bot): Dirichlet  u = T_bed
  - Top face    (z ≈ z_top): Dirichlet  u = T_nozzle
  - Side / end faces:        Robin      ∂u/∂n − μu = κ  (convection)

Previous bead (nozzle has moved away):
  - Bottom face (z ≈ z_bot): Dirichlet  u = T_bed
  - Top + side + end faces:  Robin      ∂u/∂n − μu = κ  (convection)

Robin coefficients (coordinates in mm):
  μ = −h / (k_cond × 1000)      [mm⁻¹]
  κ = −μ × T_ambient             [°C / mm]
"""

import numpy as np
import zombie

_DIM      = 3
_CHANNELS = 1
_Z_EPS_FRAC = 0.1


class ThermalSolver:

    def __init__(self,
                 T_bed:     float = 60.0,
                 T_nozzle:  float = 200.0,
                 T_ambient: float = 20.0,
                 h:         float = 25.0,    # convection coeff  W/(m²·K)
                 k_cond:    float = 0.2,     # PLA conductivity  W/(m·K)
                 n_walks:   int   = 128):
        self.T_bed     = T_bed
        self.T_nozzle  = T_nozzle
        self.T_ambient = T_ambient
        self.n_walks   = n_walks

        # Robin parameters in mm coordinates
        self.mu    = -(h / (k_cond * 1000.0))   # mm⁻¹  (negative for convection)
        self.kappa = -self.mu * T_ambient         # °C/mm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, vertices: np.ndarray, faces: np.ndarray,
              query_point: np.ndarray,
              is_current_bead: bool = True) -> float:
        """
        Estimate the steady-state temperature at *query_point* inside
        the closed bead mesh.

        vertices        : (N, 3) float array  — XYZ in mm
        faces           : (M, 3) int   array  — vertex index triples
        query_point     : (3,)   float array  — must lie inside the mesh
        is_current_bead : True  → nozzle is still on top face (Dirichlet T_nozzle)
                          False → nozzle has moved away; top face cools by convection

        Returns temperature in °C.
        """
        verts = np.asarray(vertices,    dtype=np.float32)
        faces = np.asarray(faces,       dtype=np.int32)
        qpt   = np.asarray(query_point, dtype=np.float32)

        z_bot = float(verts[:, 2].min())
        z_top = float(verts[:, 2].max())
        eps   = (z_top - z_bot) * _Z_EPS_FRAC

        # ---- zombie geometry ------------------------------------------
        positions = zombie.Float3List(verts)
        bbox      = zombie.Utils.compute_bounding_box(positions, False, 1.0,
                                                       dim=_DIM)

        absorbing_pos, absorbing_idx, reflecting_pos, reflecting_idx = \
            self._partition_mesh(verts, faces, z_bot, z_top, eps,
                                 is_current_bead)

        gq, _gq_refs = self._build_geometric_queries(
            bbox, absorbing_pos, absorbing_idx,
            reflecting_pos, reflecting_idx)

        pde, _grid_refs = self._build_pde(verts, z_bot, z_top,
                                          is_current_bead)

        # ---- query point -----------------------------------------------
        dist_abs = gq.compute_dist_to_absorbing_boundary(qpt, False)
        dist_ref = gq.compute_dist_to_reflecting_boundary(qpt, False)

        sample_pt = zombie.Solvers.SamplePoint(
            qpt, np.zeros(_DIM, dtype=np.float32),
            zombie.Solvers.SampleType.InDomain,
            zombie.Solvers.EstimationQuantity.Solution,
            1.0, dist_abs, dist_ref,
            dim=_DIM, channels=_CHANNELS)

        stats = zombie.Solvers.SampleStatistics(dim=_DIM, channels=_CHANNELS)

        sample_pts = zombie.Solvers.SamplePointList(
            [sample_pt], dim=_DIM, channels=_CHANNELS)
        stats_list = zombie.Solvers.SampleStatisticsList(
            [stats],     dim=_DIM, channels=_CHANNELS)

        # ---- solver ----------------------------------------------------
        has_reflecting = len(reflecting_idx) > 0
        walk_settings  = zombie.Solvers.WalkSettings(
            1e-3, 1e-3, 1e-3,
            0.0, np.inf,
            1024, 0, 1024,
            False,
            True, True, False,
            False,
            not has_reflecting,   # ignore reflecting contribution if none
            False,
            False)

        pb     = zombie.Utils.ProgressBar(1)
        rp     = zombie.Utils.get_report_progress_callback(pb)
        solver = zombie.Solvers.WalkOnStars(gq, dim=_DIM, channels=_CHANNELS)
        solver.solve(pde, walk_settings,
                     zombie.IntList([self.n_walks]),
                     sample_pts, stats_list,
                     True, rp)
        pb.finish()

        return float(stats_list[0].get_estimated_solution())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _partition_mesh(self, verts, faces, z_bot, z_top, eps,
                        is_current_bead):
        """
        Split faces into absorbing (Dirichlet) and reflecting (Robin).

        is_current_bead=True  → bottom + top absorbing, sides/ends reflecting
        is_current_bead=False → bottom only absorbing, top+sides/ends reflecting
        """
        centroids_z  = verts[faces, 2].mean(axis=1)
        is_bottom    = centroids_z <= z_bot + eps
        is_top       = centroids_z >= z_top - eps

        if is_current_bead:
            is_absorbing = is_bottom | is_top
        else:
            is_absorbing = is_bottom          # top now cools by convection

        absorbing_faces  = faces[is_absorbing].astype(np.int32)
        reflecting_faces = faces[~is_absorbing].astype(np.int32)

        absorbing_pos  = zombie.Float3List(verts)
        absorbing_idx  = zombie.Int3List(absorbing_faces)
        reflecting_pos = zombie.Float3List(verts)
        reflecting_idx = zombie.Int3List(reflecting_faces)

        return absorbing_pos, absorbing_idx, reflecting_pos, reflecting_idx

    def _build_pde(self, verts, z_bot, z_top, is_current_bead):
        """
        Build PDE:
          - Dirichlet grid: T_bed on bottom, T_nozzle on top (current bead only)
          - Robin on reflecting faces: ∂u/∂n − μu = κ
        """
        has_robin = True   # always have Robin on at least side/end faces

        # Dirichlet grid (3 cells in Z)
        if is_current_bead:
            # bottom → T_bed, middle → T_ambient (sides), top → T_nozzle
            buf = np.array([self.T_bed, self.T_ambient, self.T_nozzle],
                           dtype=np.float32)
        else:
            # only bottom is absorbing; use constant T_bed everywhere in grid
            buf = np.array([self.T_bed, self.T_bed, self.T_bed],
                           dtype=np.float32)

        shape = np.array([1, 1, 3], dtype=np.int32)
        d_min = np.array([verts[:, 0].min(), verts[:, 1].min(), z_bot],
                         dtype=np.float32)
        d_max = np.array([verts[:, 0].max(), verts[:, 1].max(), z_top],
                         dtype=np.float32)

        pde = zombie.Core.PDE(dim=_DIM, channels=_CHANNELS)
        pde.source    = zombie.Core.get_constant_source_callback(
                            0.0, dim=_DIM, channels=_CHANNELS)
        pde.dirichlet = zombie.Utils.get_dense_grid_dirichlet_callback(
                            buf, shape, d_min, d_max,
                            dim=_DIM, channels=_CHANNELS)
        pde.robin     = zombie.Core.get_constant_robin_callback(
                            self.kappa, dim=_DIM, channels=_CHANNELS)
        pde.robin_coeff = zombie.Core.get_constant_robin_coefficient_callback(
                            self.mu, dim=_DIM)
        pde.has_reflecting_boundary_conditions = \
            zombie.Core.get_constant_indicator_callback(has_robin, dim=_DIM)
        pde.absorption_coeff             = 0.0
        pde.are_robin_conditions_pure_neumann = False
        pde.are_robin_coeffs_nonnegative      = (self.mu >= 0)

        return pde, (buf, shape, d_min, d_max)

    def _build_geometric_queries(self, bbox,
                                  absorbing_pos, absorbing_idx,
                                  reflecting_pos, reflecting_idx):
        """Build geometric queries with Dirichlet + optional Robin handler."""
        gq = zombie.Core.GeometricQueries(
            True, bbox[0], bbox[1], dim=_DIM)

        # absorbing boundary
        dh = zombie.Utils.FcpwDirichletBoundaryHandler(dim=_DIM)
        dh.build_acceleration_structure(absorbing_pos, absorbing_idx)
        zombie.Utils.populate_geometric_queries_for_dirichlet_boundary(
            dh, gq, dim=_DIM)

        refs = [dh, absorbing_pos, absorbing_idx]

        # reflecting (Robin) boundary
        if len(reflecting_idx) > 0:
            ignore_silhouette = \
                zombie.Utils.get_ignore_candidate_silhouette_callback(False)
            branch_weight = zombie.Utils.get_branch_traversal_weight_callback()

            rh = zombie.Utils.FcpwRobinBoundaryHandler(dim=_DIM)
            n  = len(reflecting_idx)
            min_coeffs = zombie.FloatList([abs(self.mu)] * n)
            max_coeffs = zombie.FloatList([abs(self.mu)] * n)
            rh.build_acceleration_structure(
                reflecting_pos, reflecting_idx,
                ignore_silhouette, min_coeffs, max_coeffs)
            zombie.Utils.populate_geometric_queries_for_robin_boundary(
                rh, branch_weight, gq, dim=_DIM)

            refs += [rh, reflecting_pos, reflecting_idx,
                     min_coeffs, max_coeffs]

        return gq, refs