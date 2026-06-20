"""
Example 01: Plot Laplace solution over the full domain using Zombie (Walk-on-Stars)

PDE:  ∇²u = 0  on [0,1]²
BCs (all Dirichlet):
    Bottom  y=0  →  u = 0
    Left    x=0  →  u = 0
    Right   x=1  →  u = 0
    Top     y=1  →  u = 1

Plots: Analytical solution | Zombie solution | Absolute error

Run with:
    source ~/.MontoThermoPOC312/bin/activate
    python experimentDev/example02/example02.py
"""

import math
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import zombie

DIM       = 2
CHANNELS  = 1
N_WALKS   = 2000     # fewer per point — we have many grid points
MAX_STEPS = 1024
EPS       = 1e-3
N_GRID    = 25      # N_GRID × N_GRID interior points

# ── Analytical solution (Fourier series, odd terms only) ───────────────────────
def u_exact(x, y, n_terms=40):
    total = 0.0
    for k in range(n_terms):
        n = 2 * k + 1
        total += (4.0 / (n * math.pi)) * math.sin(n * math.pi * x) \
                 * math.sinh(n * math.pi * y) / math.sinh(n * math.pi)
    return total

# ── Build unit square boundary as a 2D OBJ file ───────────────────────────────
def build_unit_square_obj(n_per_edge=20):
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    vertices, edges = [], []
    for k in range(4):
        p0   = np.array(corners[k])
        p1   = np.array(corners[(k + 1) % 4])
        base = len(vertices)
        for i in range(n_per_edge):
            vertices.append(p0 + (i / n_per_edge) * (p1 - p0))
        for i in range(n_per_edge - 1):
            edges.append((base + i, base + i + 1))
        edges.append((base + n_per_edge - 1, ((k + 1) % 4) * n_per_edge))
    path = os.path.join(tempfile.gettempdir(), "unit_square.obj")
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f}\n")
        for e in edges:
            f.write(f"l {e[0] + 1} {e[1] + 1}\n")
    return path

# ── Dirichlet BC grid ──────────────────────────────────────────────────────────
def build_dirichlet_grid(res=128):
    shape  = np.array([res, res], dtype=np.int32)
    buffer = np.zeros(res * res, dtype=np.float32)
    for x_idx in range(res):
        buffer[x_idx * res + (res - 1)] = 1.0   # top edge: y=1 → u=1
    return buffer, shape

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Laplace equation – Zombie WalkOnStars  (grid solve)")
    print(f"  Grid: {N_GRID}×{N_GRID} = {N_GRID**2} points,  {N_WALKS} walks each")
    print("=" * 55)

    domain_min = np.array([0.0, 0.0])
    domain_max = np.array([1.0, 1.0])

    # 1. Boundary mesh
    obj_path  = build_unit_square_obj(n_per_edge=20)
    positions = zombie.FloatNList(dim=DIM)
    indices   = zombie.IntNList(dim=DIM)
    zombie.Utils.load_boundary_mesh(obj_path, positions, indices, dim=DIM)
    zombie.Utils.flip_orientation(indices, dim=DIM)

    # 2. PDE
    dirichlet_buffer, dirichlet_shape = build_dirichlet_grid(res=128)
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

    # 4. Build interior grid (avoid boundaries by offsetting half a cell)
    xs = np.linspace(0.0, 1.0, N_GRID + 2)[1:-1]   # interior x values
    ys = np.linspace(0.0, 1.0, N_GRID + 2)[1:-1]   # interior y values
    grid_pts = [(x, y) for y in ys for x in xs]     # row-major: y outer, x inner

    # 5. Sample points
    sample_pts_list   = []
    sample_stats_list = []
    for (x, y) in grid_pts:
        pt     = np.array([x, y])
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

    # 6. Solve
    walk_settings = zombie.Solvers.WalkSettings(
        EPS, EPS, EPS, 0.0, np.inf,
        MAX_STEPS, 0, MAX_STEPS,
        False, True, True, False,
        False, True, True, False
    )
    n_walks_list    = zombie.IntList([N_WALKS] * len(grid_pts))
    progress_bar    = zombie.Utils.ProgressBar(len(grid_pts))
    report_progress = zombie.Utils.get_report_progress_callback(progress_bar)

    solver = zombie.Solvers.WalkOnStars(geometric_queries, dim=DIM, channels=CHANNELS)
    solver.solve(pde, walk_settings, n_walks_list, sample_pts, sample_stats,
                 False, report_progress)   # False = multi-threaded
    progress_bar.finish()

    # 7. Extract results into 2D grids
    Z_zombie = np.zeros((N_GRID, N_GRID))
    Z_exact  = np.zeros((N_GRID, N_GRID))

    for idx, (x, y) in enumerate(grid_pts):
        row = idx // N_GRID   # y index
        col = idx  % N_GRID   # x index
        Z_zombie[row, col] = sample_stats[idx].get_estimated_solution()
        Z_exact [row, col] = u_exact(x, y)

    Z_error = np.abs(Z_zombie - Z_exact)

    # 8. Plot
    X, Y = np.meshgrid(xs, ys)
    vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = ["Analytical (exact)", f"Zombie WalkOnStars\n({N_WALKS} walks/point)", "Absolute error"]
    data   = [Z_exact, Z_zombie, Z_error]
    cmaps  = ["turbo", "turbo", "Reds"]

    for ax, title, Z, cmap in zip(axes, titles, data, cmaps):
        if cmap == "Reds":
            im = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
        else:
            im = ax.contourf(X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # Draw BC annotations
        ax.axhline(1.0, color="white", lw=2, ls="--")
        ax.text(0.5, 0.97, "u=1", ha="center", va="top",
                color="white", fontsize=9, transform=ax.transAxes)

    fig.suptitle("Laplace equation  ∇²u = 0  on [0,1]²   |   u=1 top,  u=0 elsewhere",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "laplace_solution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved → {out_path}")
    print(f"Max error: {Z_error.max():.4f}   Mean error: {Z_error.mean():.4f}")


main()