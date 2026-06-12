"""
Example 04: Transient heat equation on [0,1]² — Zombie Walk-on-Stars

PDE:  ∂u/∂t = α ∇²u          on  Ω = [0,1]²
BC:   u = 1  (top,  y = 1),   u = 0  (bottom, left, right)
IC:   u(x, y, 0) = 0

Time discretisation: backward Euler
    (u^{n+1} − u^n) / Δt = α ∇²u^{n+1}
  ⟹  ∇²u^{n+1} − σ u^{n+1} = −σ u^n      σ = 1/(αΔt)

Zombie solves the screened-Poisson (Yukawa) equation:
    Δu − λ u = −f
Setting  λ = σ  and  f = σ·u^n  gives the correct backward-Euler step.

At steady state (u^{n+1} ≈ u^n) the equation collapses to ∇²u = 0,
which is the example02 Laplace solution.

Run with:
    source ~/.MontoThermoPOC312/bin/activate
    python experimentDev/example04/example04.py
"""

import math
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import zombie

# ── Simulation parameters ─────────────────────────────────────────────────────
DIM       = 2
CHANNELS  = 1
N_WALKS   = 2000        # Monte-Carlo walks per interior point per time step
MAX_STEPS = 1024
EPS       = 1e-3
N_GRID    = 20         # N_GRID × N_GRID interior evaluation points

ALPHA   = 1.0          # thermal diffusivity (normalised, arbitrary units)
DT      = 0.5          # time step  (σ = 1/(α·Δt) = 2 — small enough for WoS)
N_STEPS = 6            # t_final = 3.0  (well past the transient τ ≈ 0.05)
SIGMA   = 1.0 / (ALPHA * DT)   # screened-Poisson / Yukawa coefficient

# With α=1, fundamental decay τ = 1/(α·π²·2) ≈ 0.051
# Keeping σ = 1/(α·Δt) ≤ 2 is required for WoS: boundary influence on interior
# points scales as exp(−√σ·dist), which is undetectable for σ=50 (old Δt=0.02).
# Saved times: 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 — span the full transient
SAVE_AT = {0, 1, 2, 3, 4, 6}   # steps whose snapshots will be plotted

# ── Analytical steady-state solution (same Fourier series as example02) ───────
def u_steady(x, y, n_terms=40):
    total = 0.0
    for k in range(n_terms):
        n = 2*k + 1
        total += (4.0 / (n * math.pi)) * math.sin(n * math.pi * x) \
                 * math.sinh(n * math.pi * y) / math.sinh(n * math.pi)
    return total

# ── Full transient analytical solution ────────────────────────────────────────
# Decompose: u = u_s + v  where v satisfies homogeneous BCs and v(x,y,0) = -u_s.
#
# Eigenfunctions for homogeneous Dirichlet BCs: sin(mπx) sin(nπy)
# Eigenvalues: λ_mn = π²(m²+n²)
#
# Projecting -u_s onto the eigenbasis gives C_mn (only odd m are non-zero):
#   C_mn = 8·(-1)^n · n / (m·π²·(m²+n²))     m = 1,3,5,…   n = 1,2,3,…
#
# Full solution:
#   u(x,y,t) = u_s(x,y)
#             + Σ_{m odd} Σ_{n≥1} C_mn · sin(mπx)·sin(nπy)·exp(−α π²(m²+n²) t)
def u_exact(x, y, t, n_terms_m=15, n_terms_n=30):
    """Analytical solution valid for all t ≥ 0."""
    # Steady-state part (converges quickly at all t)
    total = u_steady(x, y, n_terms=n_terms_m)
    # Transient correction (decays exponentially; needs more n-terms than m-terms)
    pi2 = math.pi ** 2
    for ki in range(n_terms_m):
        m = 2 * ki + 1   # odd m only
        for n in range(1, n_terms_n + 1):
            decay = math.exp(-ALPHA * pi2 * (m * m + n * n) * t)
            if decay < 1e-14:   # negligible — skip remaining n for this m
                break
            sign  = (-1) ** n
            coeff = 8.0 * sign * n / (m * pi2 * (m * m + n * n))
            total += coeff * math.sin(m * math.pi * x) \
                           * math.sin(n * math.pi * y) * decay
    return total

# ── Unit-square boundary mesh ─────────────────────────────────────────────────
def build_unit_square_obj(n_per_edge=20):
    corners = [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)]
    vertices, edges = [], []
    for k in range(4):
        p0 = np.array(corners[k]); p1 = np.array(corners[(k+1)%4])
        base = len(vertices)
        for i in range(n_per_edge):
            vertices.append(p0 + (i / n_per_edge) * (p1 - p0))
        for i in range(n_per_edge - 1):
            edges.append((base+i, base+i+1))
        edges.append((base + n_per_edge - 1, ((k+1) % 4) * n_per_edge))
    path = os.path.join(tempfile.gettempdir(), "unit_square.obj")
    with open(path, "w") as f:
        for v in vertices: f.write(f"v {v[0]:.6f} {v[1]:.6f}\n")
        for e in edges:    f.write(f"l {e[0]+1} {e[1]+1}\n")
    return path

# ── Dirichlet BC grid: u=1 at top (y=1), u=0 elsewhere ───────────────────────
def build_dirichlet_grid(res=128):
    shape  = np.array([res, res], dtype=np.int32)
    buffer = np.zeros(res * res, dtype=np.float32)
    for xi in range(res):
        buffer[xi * res + (res - 1)] = 1.0   # yi = res-1 → y = 1 → u = 1
    return buffer, shape

# ── Source buffer: f = σ · u^n  on a full (M×M) grid ─────────────────────────
def build_source_buffer(u_current):
    """
    u_current : shape (N_GRID, N_GRID),  layout [row = y_idx, col = x_idx].

    Returns a flat float32 buffer with shape M × M  (M = N_GRID + 2)
    that covers [0,1]² uniformly — interior values come from u_current,
    boundary values are the Dirichlet BCs (u=1 at top, u=0 elsewhere).

    Buffer layout (same convention as build_dirichlet_grid):
        buffer[xi * M + yi],  xi = x-index,  yi = y-index.
    """
    M = N_GRID + 2
    u_full = np.zeros((M, M), dtype=np.float32)  # [xi, yi]
    u_full[:, M - 1] = 1.0   # top edge (y = 1) → u = 1

    # Map interior: u_current[y_idx, x_idx] → u_full[x_idx+1, y_idx+1]
    for y_idx in range(N_GRID):
        for x_idx in range(N_GRID):
            u_full[x_idx + 1, y_idx + 1] = float(u_current[y_idx, x_idx])

    # Zombie solves Δu − λu = −f, so we pass f = σ·u^n (positive)
    source = (SIGMA * u_full).flatten().astype(np.float32)
    return source, np.array([M, M], dtype=np.int32)

# ── One-time geometry / geometric-queries setup ───────────────────────────────
def setup_geometry(domain_min, domain_max):
    """
    Returns (geometric_queries, dirichlet_handler, neumann_handler).

    IMPORTANT: the two handlers MUST be kept alive (held in a variable) for as
    long as geometric_queries is used.  populate_geometric_queries_* stores raw
    C++ pointers into the handlers' BVH data; if the handlers are garbage-
    collected the pointers dangle and the process crashes with SIGSEGV.
    """
    obj_path  = build_unit_square_obj(n_per_edge=20)
    positions = zombie.FloatNList(dim=DIM)
    indices   = zombie.IntNList(dim=DIM)
    zombie.Utils.load_boundary_mesh(obj_path, positions, indices, dim=DIM)
    zombie.Utils.flip_orientation(indices, dim=DIM)

    geometric_queries = zombie.Core.GeometricQueries(
        True, domain_min, domain_max, dim=DIM)

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

    return geometric_queries, dirichlet_handler, neumann_handler

# ── Build PDE object for the current time step ────────────────────────────────
def build_pde(u_current, domain_min, domain_max, dirichlet_buffer, dirichlet_shape):
    source_buf, source_shape = build_source_buffer(u_current)

    pde = zombie.Core.PDE(dim=DIM, channels=CHANNELS)
    pde.absorption_coeff                  = SIGMA
    pde.are_robin_conditions_pure_neumann = True
    pde.are_robin_coeffs_nonnegative      = True
    pde.source = zombie.Utils.get_dense_grid_source_callback(
        source_buf, source_shape, domain_min, domain_max,
        dim=DIM, channels=CHANNELS)
    pde.dirichlet = zombie.Utils.get_dense_grid_dirichlet_callback(
        dirichlet_buffer, dirichlet_shape, domain_min, domain_max,
        dim=DIM, channels=CHANNELS)
    pde.robin       = zombie.Core.get_constant_robin_callback(
        0.0, dim=DIM, channels=CHANNELS)
    pde.robin_coeff = zombie.Core.get_constant_robin_coefficient_callback(
        0.0, dim=DIM)
    pde.has_reflecting_boundary_conditions = \
        zombie.Core.get_constant_indicator_callback(False, dim=DIM)
    return pde

# ── Run WoS for one time step, return N_GRID × N_GRID solution grid ───────────
def solve_step(pde, geometric_queries, xs, ys, walk_settings):
    grid_pts = [(x, y) for y in ys for x in xs]   # row-major: y outer, x inner

    sample_pts_list, sample_stats_list = [], []
    for (x, y) in grid_pts:
        pt    = np.array([x, y])
        d_abs = geometric_queries.compute_dist_to_absorbing_boundary(pt, False)
        d_ref = geometric_queries.compute_dist_to_reflecting_boundary(pt, False)
        sp = zombie.Solvers.SamplePoint(
            pt, np.zeros(DIM),
            zombie.Solvers.SampleType.InDomain,
            zombie.Solvers.EstimationQuantity.Solution,
            1.0, d_abs, d_ref,
            dim=DIM, channels=CHANNELS)
        ss = zombie.Solvers.SampleStatistics(dim=DIM, channels=CHANNELS)
        sample_pts_list.append(sp)
        sample_stats_list.append(ss)

    sample_pts   = zombie.Solvers.SamplePointList(
        sample_pts_list,   dim=DIM, channels=CHANNELS)
    sample_stats = zombie.Solvers.SampleStatisticsList(
        sample_stats_list, dim=DIM, channels=CHANNELS)

    n_walks_list    = zombie.IntList([N_WALKS] * len(grid_pts))
    progress_bar    = zombie.Utils.ProgressBar(len(grid_pts))
    report_progress = zombie.Utils.get_report_progress_callback(progress_bar)

    solver = zombie.Solvers.WalkOnStars(geometric_queries, dim=DIM, channels=CHANNELS)
    solver.solve(pde, walk_settings, n_walks_list, sample_pts, sample_stats,
                 False, report_progress)
    progress_bar.finish()

    Z = np.zeros((N_GRID, N_GRID))
    for idx in range(len(grid_pts)):
        Z[idx // N_GRID, idx % N_GRID] = \
            sample_stats[idx].get_estimated_solution()
    return Z

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Transient heat equation – Zombie WalkOnStars (2D)")
    print(f"  α = {ALPHA},   Δt = {DT},   σ = 1/(αΔt) = {SIGMA:.1f}")
    print(f"  {N_STEPS} steps,   t_final = {N_STEPS * DT:.2f}")
    print(f"  Grid: {N_GRID}×{N_GRID},   {N_WALKS} walks/point/step")
    print("=" * 65)

    domain_min = np.array([0.0, 0.0])
    domain_max = np.array([1.0, 1.0])

    dirichlet_buffer, dirichlet_shape = build_dirichlet_grid(res=128)
    # Keep handlers alive — geometric_queries holds raw C++ pointers into them
    geometric_queries, dirichlet_handler, neumann_handler = \
        setup_geometry(domain_min, domain_max)

    xs = np.linspace(0.0, 1.0, N_GRID + 2)[1:-1]
    ys = np.linspace(0.0, 1.0, N_GRID + 2)[1:-1]

    walk_settings = zombie.Solvers.WalkSettings(
        EPS, EPS, EPS, 0.0, np.inf,
        MAX_STEPS, 0, MAX_STEPS,
        False, True, True, False,
        False, True, True, False
    )

    # ── Time loop ─────────────────────────────────────────────────────────────
    u_current = np.zeros((N_GRID, N_GRID))   # IC: u(x, y, 0) = 0
    snapshots = {0: u_current.copy()}

    for step in range(1, N_STEPS + 1):
        print(f"\n--- Step {step:3d}/{N_STEPS}   t = {step * DT:.3f} ---")
        pde       = build_pde(u_current, domain_min, domain_max,
                              dirichlet_buffer, dirichlet_shape)
        u_current = solve_step(pde, geometric_queries, xs, ys, walk_settings)
        if step in SAVE_AT:
            snapshots[step] = u_current.copy()

    # ── Analytical solution and error at each saved step ─────────────────────
    X, Y = np.meshgrid(xs, ys)

    t_final = N_STEPS * DT
    Z_exact_final = np.vectorize(lambda x, y: u_exact(x, y, t_final))(X, Y)
    err = np.abs(u_current - Z_exact_final)
    print(f"\nFinal (t={t_final:.2f}) vs analytical:  "
          f"max_err = {err.max():.4f},  mean_err = {err.mean():.4f}")

    # ── Plot: WoS snapshots (top row) vs analytical (bottom row) ─────────────
    steps_to_plot = sorted(snapshots.keys())
    ncols = len(steps_to_plot)

    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))

    for col, step in enumerate(steps_to_plot):
        t_val = step * DT
        Z_wos = snapshots[step]
        Z_ana = np.vectorize(lambda x, y: u_exact(x, y, t_val))(X, Y)

        # Shared color range for this column: min/max across both solutions
        vmin = float(min(Z_wos.min(), Z_ana.min()))
        vmax = float(max(Z_wos.max(), Z_ana.max()))
        levels = np.linspace(vmin, vmax, 21)

        # Top row: WoS solution
        im = axes[0, col].contourf(X, Y, Z_wos, levels=levels, cmap="turbo")
        plt.colorbar(im, ax=axes[0, col])
        axes[0, col].set_title(f"WoS   t = {t_val:.2f}", fontsize=9)
        axes[0, col].set_xlabel("x"); axes[0, col].set_ylabel("y")

        # Bottom row: analytical solution
        im = axes[1, col].contourf(X, Y, Z_ana, levels=levels, cmap="turbo")
        plt.colorbar(im, ax=axes[1, col])
        axes[1, col].set_title(f"Analytical   t = {t_val:.2f}", fontsize=9)
        axes[1, col].set_xlabel("x"); axes[1, col].set_ylabel("y")

    fig.suptitle(
        f"Transient heat  ∂u/∂t = α∇²u  on [0,1]²   |   u=1 top, u=0 elsewhere\n"
        f"α = {ALPHA},   Δt = {DT},   σ = {SIGMA:.1f}   ({N_WALKS} walks/point)\n"
        f"Top: Zombie WoS  |  Bottom: exact Fourier series",
        fontsize=11, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "transient_solution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_path}")

    # ── 1D comparison: one subplot per saved time step, u vs y at x=0.5 ───────
    x_fixed  = 0.5
    x_col    = int(np.argmin(np.abs(xs - x_fixed)))
    x_actual = xs[x_col]
    y_fine   = np.linspace(0.0, 1.0, 200)

    ncols1d = len(steps_to_plot)
    fig1d, axes1d = plt.subplots(1, ncols1d, figsize=(4 * ncols1d, 4),
                                  sharey=True)

    for ax, step in zip(axes1d, steps_to_plot):
        t_val = step * DT

        u_wos = snapshots[step][:, x_col]
        u_ana = np.array([u_exact(x_actual, yv, t_val) for yv in y_fine])

        ax.plot(y_fine, u_ana, color="steelblue", lw=2, label="Analytical")
        ax.plot(ys, u_wos, color="crimson", lw=0,
                marker="o", ms=5, label="WoS")

        ax.set_title(f"t = {t_val:.2f}", fontsize=10)
        ax.set_xlabel("y", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes1d[0].set_ylabel(f"u(x={x_actual:.3f},  y,  t)", fontsize=10)

    fig1d.suptitle(
        f"1D profile at x = {x_actual:.3f}  —  solid: analytical,  dots: WoS",
        fontsize=11)
    plt.tight_layout()
    out_path_1d = os.path.join(os.path.dirname(__file__), "transient_1d_profile.png")
    plt.savefig(out_path_1d, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_path_1d}")

    # ── Save snapshots for comparison in example05 ────────────────────────────
    save_dict = {'xs': xs, 'ys': ys, 'DT': np.float64(DT)}
    for step in sorted(snapshots.keys()):
        save_dict[f'step_{step}'] = snapshots[step]
    out_npz = os.path.join(os.path.dirname(__file__), 'wos_snapshots.npz')
    np.savez(out_npz, **save_dict)
    print(f"WoS snapshots saved → {out_npz}")


main()