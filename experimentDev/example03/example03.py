"""
Example 03: Solve Laplace equation on a 3D unit cube using Zombie (Walk-on-Stars)

PDE:  ∇²u = 0  on [0,1]³
BCs (all Dirichlet):
    Top    z=1  →  u = 1
    All other faces  →  u = 0

Plots: 3×3 grid — three z-slices (z=0.25, 0.50, 0.75)
       each showing: Analytical | Zombie | Absolute error

Run with:
    source ~/.MontoThermoPOC312/bin/activate
    python experimentDev/example03/example03.py
"""

import math
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3D projection
import zombie

DIM      = 3
CHANNELS = 1
N_WALKS  = 2000
MAX_STEPS = 1024
EPS       = 1e-3
N_GRID    = 15      # N_GRID × N_GRID per z-slice
Z_SLICES  = [0.25, 0.50, 0.75]

# ── Analytical solution (double Fourier series) ────────────────────────────────
# u(x,y,z) = Σ_{m,n odd} A_mn * sin(mπx) * sin(nπy) * sinh(γ_mn z) / sinh(γ_mn)
# where γ_mn = π√(m²+n²),  A_mn = 16 / (mn π²)
def u_exact(x, y, z, n_terms=8):
    total = 0.0
    for ki in range(n_terms):
        m = 2 * ki + 1
        for kj in range(n_terms):
            n = 2 * kj + 1
            gamma = math.pi * math.sqrt(m * m + n * n)
            A = 16.0 / (m * n * math.pi ** 2)
            total += A * math.sin(m * math.pi * x) \
                       * math.sin(n * math.pi * y) \
                       * math.sinh(gamma * z) / math.sinh(gamma)
    return total

# ── Build triangulated unit cube OBJ ──────────────────────────────────────────
def build_unit_cube_obj(n_per_edge=6):
    """
    Creates a triangulated OBJ for [0,1]^3.
    Each face is subdivided into n_per_edge × n_per_edge quads → 2 triangles each.
    Returns path to the temporary OBJ file.
    """
    vertices = []
    triangles = []

    def add_face(origin, u_vec, v_vec):
        """Add a quad face subdivided into triangles."""
        base = len(vertices)
        # Create (n+1)×(n+1) vertex grid
        for vi in range(n_per_edge + 1):
            for ui in range(n_per_edge + 1):
                pt = origin + (ui / n_per_edge) * u_vec + (vi / n_per_edge) * v_vec
                vertices.append(pt)
        # Triangulate
        stride = n_per_edge + 1
        for vi in range(n_per_edge):
            for ui in range(n_per_edge):
                i00 = base + vi * stride + ui
                i10 = base + vi * stride + ui + 1
                i01 = base + (vi + 1) * stride + ui
                i11 = base + (vi + 1) * stride + ui + 1
                triangles.append((i00, i10, i11))
                triangles.append((i00, i11, i01))

    o = np.zeros(3)
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])

    add_face(o,            ex,  ey)          # bottom z=0
    add_face(ez,           ex,  ey)          # top    z=1
    add_face(o,            ex,  ez)          # front  y=0
    add_face(ey,           ex,  ez)          # back   y=1
    add_face(o,            ey,  ez)          # left   x=0
    add_face(ex,           ey,  ez)          # right  x=1

    path = os.path.join(tempfile.gettempdir(), "unit_cube.obj")
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in triangles:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    return path

# ── Dirichlet BC volume grid ───────────────────────────────────────────────────
def build_dirichlet_grid_3d(res=32):
    """
    3D grid: shape=[res, res, res], layout: buffer[xi*res*res + yi*res + zi]
    Top face (z=1, zi = res-1) → u = 1; everything else → 0.
    """
    shape  = np.array([res, res, res], dtype=np.int32)
    buffer = np.zeros(res ** 3, dtype=np.float32)
    for xi in range(res):
        for yi in range(res):
            buffer[xi * res * res + yi * res + (res - 1)] = 1.0
    return buffer, shape

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  3D Laplace equation – Zombie WalkOnStars")
    print(f"  Slices: z = {Z_SLICES},  grid {N_GRID}×{N_GRID} each,  {N_WALKS} walks/point")
    print("=" * 60)

    domain_min = np.array([0.0, 0.0, 0.0])
    domain_max = np.array([1.0, 1.0, 1.0])

    # 1. Boundary mesh
    obj_path  = build_unit_cube_obj(n_per_edge=6)
    positions = zombie.FloatNList(dim=DIM)
    indices   = zombie.IntNList(dim=DIM)
    zombie.Utils.load_boundary_mesh(obj_path, positions, indices, dim=DIM)
    zombie.Utils.flip_orientation(indices, dim=DIM)

    # 2. PDE
    dirichlet_buffer, dirichlet_shape = build_dirichlet_grid_3d(res=32)
    pde = zombie.Core.PDE(dim=DIM, channels=CHANNELS)
    pde.absorption_coeff                  = 0.0
    pde.are_robin_conditions_pure_neumann = True
    pde.are_robin_coeffs_nonnegative      = True
    pde.source    = zombie.Core.get_constant_source_callback(0.0, dim=DIM, channels=CHANNELS)
    pde.dirichlet = zombie.Utils.get_dense_grid_dirichlet_callback(
                        dirichlet_buffer, dirichlet_shape,
                        domain_min, domain_max, dim=DIM, channels=CHANNELS)
    pde.robin     = zombie.Core.get_constant_robin_callback(0.0, dim=DIM, channels=CHANNELS)
    pde.robin_coeff = zombie.Core.get_constant_robin_coefficient_callback(0.0, dim=DIM)
    pde.has_reflecting_boundary_conditions = \
        zombie.Core.get_constant_indicator_callback(False, dim=DIM)

    # 3. Geometric queries
    geometric_queries = zombie.Core.GeometricQueries(True, domain_min, domain_max, dim=DIM)
    dirichlet_handler = zombie.Utils.FcpwDirichletBoundaryHandler(dim=DIM)
    dirichlet_handler.build_acceleration_structure(positions, indices)
    zombie.Utils.populate_geometric_queries_for_dirichlet_boundary(
        dirichlet_handler, geometric_queries, dim=DIM)
    empty_pos = zombie.FloatNList(dim=DIM)
    empty_idx = zombie.IntNList(dim=DIM)
    neumann_handler = zombie.Utils.FcpwNeumannBoundaryHandler(dim=DIM)
    neumann_handler.build_acceleration_structure(
        empty_pos, empty_idx,
        zombie.Utils.get_ignore_candidate_silhouette_callback(False))
    zombie.Utils.populate_geometric_queries_for_neumann_boundary(
        neumann_handler, zombie.Utils.get_branch_traversal_weight_callback(),
        geometric_queries, dim=DIM)

    # 4. Interior grid for each z-slice
    xs = np.linspace(0.0, 1.0, N_GRID + 2)[1:-1]
    ys = np.linspace(0.0, 1.0, N_GRID + 2)[1:-1]

    walk_settings = zombie.Solvers.WalkSettings(
        EPS, EPS, EPS, 0.0, np.inf,
        MAX_STEPS, 0, MAX_STEPS,
        False, True, True, False,
        False, True, True, False
    )

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    col_titles = ["Analytical (exact)", f"Zombie WalkOnStars\n({N_WALKS} walks/point)", "Absolute error"]
    cmaps = ["turbo", "turbo", "Reds"]

    # Accumulate data for the 3D plot
    slices_exact  = []
    slices_zombie = []

    for row_idx, z_val in enumerate(Z_SLICES):
        print(f"\nSolving slice z = {z_val} ...")

        grid_pts = [(x, y, z_val) for y in ys for x in xs]

        # Build sample points
        sample_pts_list   = []
        sample_stats_list = []
        for (x, y, z) in grid_pts:
            pt     = np.array([x, y, z])
            d_abs  = geometric_queries.compute_dist_to_absorbing_boundary(pt, False)
            d_refl = geometric_queries.compute_dist_to_reflecting_boundary(pt, False)
            sp = zombie.Solvers.SamplePoint(
                pt, np.zeros(DIM),
                zombie.Solvers.SampleType.InDomain,
                zombie.Solvers.EstimationQuantity.Solution,
                1.0, d_abs, d_refl,
                dim=DIM, channels=CHANNELS)
            ss = zombie.Solvers.SampleStatistics(dim=DIM, channels=CHANNELS)
            sample_pts_list.append(sp)
            sample_stats_list.append(ss)

        sample_pts   = zombie.Solvers.SamplePointList(sample_pts_list,   dim=DIM, channels=CHANNELS)
        sample_stats = zombie.Solvers.SampleStatisticsList(sample_stats_list, dim=DIM, channels=CHANNELS)

        n_walks_list    = zombie.IntList([N_WALKS] * len(grid_pts))
        progress_bar    = zombie.Utils.ProgressBar(len(grid_pts))
        report_progress = zombie.Utils.get_report_progress_callback(progress_bar)

        solver = zombie.Solvers.WalkOnStars(geometric_queries, dim=DIM, channels=CHANNELS)
        solver.solve(pde, walk_settings, n_walks_list, sample_pts, sample_stats,
                     False, report_progress)
        progress_bar.finish()

        # Extract into 2D grids
        Z_zombie = np.zeros((N_GRID, N_GRID))
        Z_exact  = np.zeros((N_GRID, N_GRID))
        for idx, (x, y, _) in enumerate(grid_pts):
            row = idx // N_GRID
            col = idx  % N_GRID
            Z_zombie[row, col] = sample_stats[idx].get_estimated_solution()
            Z_exact [row, col] = u_exact(x, y, z_val)

        Z_error = np.abs(Z_zombie - Z_exact)
        X, Y = np.meshgrid(xs, ys)

        slices_exact.append((z_val, X, Y, Z_exact))
        slices_zombie.append((z_val, X, Y, Z_zombie))

        print(f"  Max error: {Z_error.max():.4f}   Mean error: {Z_error.mean():.4f}")

        # Plot row
        vmin, vmax = 0.0, 1.0
        for col_idx, (Z, cmap) in enumerate(zip([Z_exact, Z_zombie, Z_error], cmaps)):
            ax = axes[row_idx, col_idx]
            if cmap == "Reds":
                im = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
            else:
                im = ax.contourf(X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10)
            ax.set_title(f"z = {z_val}" + (f"\n{col_titles[col_idx]}" if row_idx == 0 else ""),
                         fontsize=9)

        # Label z-slice on left axis
        axes[row_idx, 0].set_ylabel(f"z = {z_val}\ny", fontsize=9)

    fig.suptitle("3D Laplace  ∇²u = 0  on [0,1]³   |   u=1 at z=1 top,  u=0 elsewhere",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "laplace_3d_solution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved → {out_path}")

    # ── 3D figure: horizontal slices coloured by solution value ──────────────
    fig3d, axes3d = plt.subplots(1, 2, figsize=(14, 6),
                                 subplot_kw={"projection": "3d"})

    for ax3d, slices, label in zip(axes3d,
                                   [slices_exact, slices_zombie],
                                   ["Analytical (exact)",
                                    f"Zombie WalkOnStars ({N_WALKS} walks/pt)"]):
        for z_val, X, Y, Z_vals in slices:
            Z_flat = np.full_like(X, z_val)       # flat plane at height z_val
            ax3d.plot_surface(
                X, Y, Z_flat,
                facecolors=plt.get_cmap("turbo")(Z_vals),  # colour each cell by u value
                rstride=1, cstride=1,
                linewidth=0, antialiased=False,
                alpha=0.85,
                label=f"z={z_val}"
            )
            # Contour lines on each slice for clarity
            ax3d.contour(X, Y, Z_vals, zdir="z", offset=z_val,
                         levels=10, cmap="turbo", linewidths=0.6)

        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
        ax3d.set_xlim(0, 1);  ax3d.set_ylim(0, 1);  ax3d.set_zlim(0, 1)
        ax3d.set_title(label, fontsize=10)
        ax3d.view_init(elev=28, azim=-55)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig3d.colorbar(sm, ax=axes3d.tolist(), shrink=0.6, label="u  (temperature)")

    fig3d.suptitle("3D Laplace – horizontal slices at z = 0.25, 0.50, 0.75",
                   fontsize=12)
    out_path_3d = os.path.join(os.path.dirname(__file__), "laplace_3d_plot.png")
    plt.savefig(out_path_3d, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_path_3d}")

    # ── 1D verification: u(x) along y=0.5, z=0.5 ────────────────────────────
    # ys = linspace(0,1,N_GRID+2)[1:-1]  → index N_GRID//2 is exactly y=0.5
    y_mid_idx = N_GRID // 2          # index 7 for N_GRID=15 → y = 8/16 = 0.5
    y_mid     = ys[y_mid_idx]        # should be 0.5

    # Pull the z=0.5 slice (index 1 in Z_SLICES)
    z_idx = Z_SLICES.index(0.50)
    _, _, _, Z_exact_05  = slices_exact [z_idx]
    _, _, _, Z_zombie_05 = slices_zombie[z_idx]

    u_exact_line  = Z_exact_05 [y_mid_idx, :]   # shape (N_GRID,) — x varies
    u_zombie_line = Z_zombie_05[y_mid_idx, :]

    fig1d, ax1d = plt.subplots(figsize=(8, 4))
    ax1d.plot(xs, u_exact_line,  "k--",  lw=2,   label="Analytical")
    ax1d.plot(xs, u_zombie_line, "r-o",  lw=1.5, ms=5, label=f"Zombie WoS ({N_WALKS} walks)")
    ax1d.set_xlabel("x", fontsize=12)
    ax1d.set_ylabel("u(x,  y=0.5,  z=0.5)", fontsize=12)
    ax1d.set_title(f"1D cross-section:  y = {y_mid:.3f},  z = 0.50", fontsize=12)
    ax1d.legend(fontsize=11)
    ax1d.set_xlim(0, 1)
    ax1d.set_ylim(0, None)
    ax1d.grid(True, alpha=0.3)

    out_path_1d = os.path.join(os.path.dirname(__file__), "laplace_1d_crosssection.png")
    plt.tight_layout()
    plt.savefig(out_path_1d, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_path_1d}")


main()