"""
diagnostic.py — Two experiments to identify the root cause of example04 error.

Experiment 1 — small σ:  fix N_WALKS=2000, reduce σ from 50 to 2 (Δt=0.5).
Experiment 2 — more walks: keep σ=50 (Δt=0.02), increase N_WALKS to 20000.

Both are compared against the analytical solution and the original example04
result at the same final time, using a 1D profile at x=0.5.

Run with:
    source ~/.MontoThermoPOC312/bin/activate
    python experimentDev/example04/diagnostic.py
"""

import math, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
import zombie

# ── Shared analytical solution (identical to example04) ──────────────────────
ALPHA = 1.0

def u_steady(x, y, n_terms=40):
    total = 0.0
    for k in range(n_terms):
        n = 2 * k + 1
        total += (4.0 / (n * math.pi)) * math.sin(n * math.pi * x) \
                 * math.sinh(n * math.pi * y) / math.sinh(n * math.pi)
    return total

def u_exact(x, y, t, n_terms_m=15, n_terms_n=30):
    total = u_steady(x, y, n_terms=n_terms_m)
    pi2 = math.pi ** 2
    for ki in range(n_terms_m):
        m = 2 * ki + 1
        for n in range(1, n_terms_n + 1):
            decay = math.exp(-ALPHA * pi2 * (m * m + n * n) * t)
            if decay < 1e-14:
                break
            sign  = (-1) ** n
            coeff = 8.0 * sign * n / (m * pi2 * (m * m + n * n))
            total += coeff * math.sin(m * math.pi * x) \
                           * math.sin(n * math.pi * y) * decay
    return total

# ── Boundary mesh (unit square) ───────────────────────────────────────────────
def build_unit_square_obj(n_per_edge=20):
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    vertices, edges = [], []
    for k in range(4):
        p0 = np.array(corners[k]); p1 = np.array(corners[(k + 1) % 4])
        base = len(vertices)
        for i in range(n_per_edge):
            vertices.append(p0 + (i / n_per_edge) * (p1 - p0))
        for i in range(n_per_edge - 1):
            edges.append((base + i, base + i + 1))
        edges.append((base + n_per_edge - 1, ((k + 1) % 4) * n_per_edge))
    path = os.path.join(tempfile.gettempdir(), "unit_square_diag.obj")
    with open(path, "w") as f:
        for v in vertices: f.write(f"v {v[0]:.6f} {v[1]:.6f}\n")
        for e in edges:    f.write(f"l {e[0]+1} {e[1]+1}\n")
    return path

def build_dirichlet_grid(res=128):
    shape  = np.array([res, res], dtype=np.int32)
    buffer = np.zeros(res * res, dtype=np.float32)
    for xi in range(res):
        buffer[xi * res + (res - 1)] = 1.0
    return buffer, shape

def build_source_buffer(u_current, n_grid, sigma):
    M      = n_grid + 2
    u_full = np.zeros((M, M), dtype=np.float32)
    u_full[:, M - 1] = 1.0
    for y_idx in range(n_grid):
        for x_idx in range(n_grid):
            u_full[x_idx + 1, y_idx + 1] = float(u_current[y_idx, x_idx])
    source = (sigma * u_full).flatten().astype(np.float32)
    return source, np.array([M, M], dtype=np.int32)

def setup_geometry(domain_min, domain_max, dim=2):
    obj_path  = build_unit_square_obj(n_per_edge=20)
    positions = zombie.FloatNList(dim=dim)
    indices   = zombie.IntNList(dim=dim)
    zombie.Utils.load_boundary_mesh(obj_path, positions, indices, dim=dim)
    zombie.Utils.flip_orientation(indices, dim=dim)
    gq = zombie.Core.GeometricQueries(True, domain_min, domain_max, dim=dim)
    dh = zombie.Utils.FcpwDirichletBoundaryHandler(dim=dim)
    dh.build_acceleration_structure(positions, indices)
    zombie.Utils.populate_geometric_queries_for_dirichlet_boundary(dh, gq, dim=dim)
    empty_pos = zombie.FloatNList(dim=dim)
    empty_idx = zombie.IntNList(dim=dim)
    nh = zombie.Utils.FcpwNeumannBoundaryHandler(dim=dim)
    nh.build_acceleration_structure(
        empty_pos, empty_idx,
        zombie.Utils.get_ignore_candidate_silhouette_callback(False))
    zombie.Utils.populate_geometric_queries_for_neumann_boundary(
        nh, zombie.Utils.get_branch_traversal_weight_callback(), gq, dim=dim)
    return gq, dh, nh

def run_wos(n_steps, dt, n_walks, n_grid=15, dim=2, channels=1,
            eps=1e-3, max_steps=1024):
    sigma      = 1.0 / (ALPHA * dt)
    domain_min = np.array([0.0, 0.0])
    domain_max = np.array([1.0, 1.0])

    dirichlet_buf, dirichlet_shape = build_dirichlet_grid(res=128)
    gq, dh, nh = setup_geometry(domain_min, domain_max, dim=dim)

    xs = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]
    ys = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]

    walk_settings = zombie.Solvers.WalkSettings(
        eps, eps, eps, 0.0, np.inf,
        max_steps, 0, max_steps,
        False, True, True, False,
        False, True, True, False
    )

    u_current = np.zeros((n_grid, n_grid))

    for step in range(1, n_steps + 1):
        source_buf, source_shape = build_source_buffer(u_current, n_grid, sigma)

        pde = zombie.Core.PDE(dim=dim, channels=channels)
        pde.absorption_coeff                  = sigma
        pde.are_robin_conditions_pure_neumann = True
        pde.are_robin_coeffs_nonnegative      = True
        pde.source = zombie.Utils.get_dense_grid_source_callback(
            source_buf, source_shape, domain_min, domain_max,
            dim=dim, channels=channels)
        pde.dirichlet = zombie.Utils.get_dense_grid_dirichlet_callback(
            dirichlet_buf, dirichlet_shape, domain_min, domain_max,
            dim=dim, channels=channels)
        pde.robin       = zombie.Core.get_constant_robin_callback(0.0, dim=dim, channels=channels)
        pde.robin_coeff = zombie.Core.get_constant_robin_coefficient_callback(0.0, dim=dim)
        pde.has_reflecting_boundary_conditions = \
            zombie.Core.get_constant_indicator_callback(False, dim=dim)

        grid_pts = [(x, y) for y in ys for x in xs]
        sp_list, ss_list = [], []
        for (x, y) in grid_pts:
            pt    = np.array([x, y])
            d_abs = gq.compute_dist_to_absorbing_boundary(pt, False)
            d_ref = gq.compute_dist_to_reflecting_boundary(pt, False)
            sp = zombie.Solvers.SamplePoint(
                pt, np.zeros(dim),
                zombie.Solvers.SampleType.InDomain,
                zombie.Solvers.EstimationQuantity.Solution,
                1.0, d_abs, d_ref, dim=dim, channels=channels)
            ss = zombie.Solvers.SampleStatistics(dim=dim, channels=channels)
            sp_list.append(sp); ss_list.append(ss)

        sample_pts   = zombie.Solvers.SamplePointList(sp_list,   dim=dim, channels=channels)
        sample_stats = zombie.Solvers.SampleStatisticsList(ss_list, dim=dim, channels=channels)
        n_walks_list = zombie.IntList([n_walks] * len(grid_pts))
        pb           = zombie.Utils.ProgressBar(len(grid_pts))
        solver       = zombie.Solvers.WalkOnStars(gq, dim=dim, channels=channels)
        solver.solve(pde, walk_settings, n_walks_list, sample_pts, sample_stats,
                     False, zombie.Utils.get_report_progress_callback(pb))
        pb.finish()

        Z = np.zeros((n_grid, n_grid))
        for idx in range(len(grid_pts)):
            Z[idx // n_grid, idx % n_grid] = \
                sample_stats[idx].get_estimated_solution()
        u_current = Z
        print(f"  step {step}/{n_steps}  dt={dt}  σ={sigma:.1f}  n_walks={n_walks}  "
              f"max(u)={u_current.max():.4f}")

    return u_current, xs, ys, n_steps * dt


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: small σ — dt=0.5, n_steps=2, n_walks=2000
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Experiment 1: small σ  (dt=0.5, σ=2, 2 steps, 2000 walks)")
print("=" * 60)
u_exp1, xs1, ys1, t_exp1 = run_wos(n_steps=2, dt=0.5, n_walks=2000, n_grid=15)

# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: more walks — dt=0.02, n_steps=4, n_walks=20000
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Experiment 2: more walks  (dt=0.02, σ=50, 4 steps, 20000 walks)")
print("=" * 60)
u_exp2, xs2, ys2, t_exp2 = run_wos(n_steps=4, dt=0.02, n_walks=20000, n_grid=15)

# ── Original example04 result (re-run 4 steps, 2000 walks, σ=50) ─────────────
print("\n" + "=" * 60)
print("  Original: dt=0.02, σ=50, 4 steps, 2000 walks")
print("=" * 60)
u_orig, xs0, ys0, t_orig = run_wos(n_steps=4, dt=0.02, n_walks=2000, n_grid=15)

# ══════════════════════════════════════════════════════════════════════════════
# Plot: 1D profile at x ≈ 0.5 for all three cases
# ══════════════════════════════════════════════════════════════════════════════
x_fixed = 0.5
y_fine  = np.linspace(0.0, 1.0, 300)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Panel A: Experiment 1 vs Original (both at t_exp1) ───────────────────────
ax = axes[0]

# Analytical at t_exp1 = 1.0 s
u_ana1 = np.array([u_exact(x_fixed, yv, t_exp1) for yv in y_fine])
ax.plot(y_fine, u_ana1, 'k-', lw=2, label=f'Analytical  t={t_exp1:.2f}s')

# Exp1: small σ (dt=0.5, σ=2)
x_col1   = int(np.argmin(np.abs(xs1 - x_fixed)))
u_wos1   = u_exp1[:, x_col1]
sigma1   = 1.0 / (ALPHA * 0.5)
ax.plot(ys1, u_wos1, 'o', color='steelblue', ms=7,
        label=f'WoS  σ={sigma1:.0f}  (dt=0.5, 2 steps, 2000 walks)')

# Original at same t (need to re-run 50 steps with dt=0.02 to reach t=1.0,
# or show at the closest matching t — use 4 steps dt=0.02 → t=0.08 with separate analytical)
x_col0 = int(np.argmin(np.abs(xs0 - x_fixed)))
u_wos0 = u_orig[:, x_col0]
u_ana0 = np.array([u_exact(x_fixed, yv, t_orig) for yv in y_fine])
ax.plot(y_fine, u_ana0, 'k--', lw=1.5, alpha=0.5,
        label=f'Analytical  t={t_orig:.2f}s')
ax.plot(ys0, u_wos0, 's', color='tomato', ms=6,
        label=f'WoS  σ=50  (dt=0.02, 4 steps, 2000 walks)')

ax.set_xlabel('y', fontsize=12)
ax.set_ylabel(f'u(x={x_fixed}, y, t)', fontsize=12)
ax.set_title('Experiment 1: Does reducing σ help?', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.grid(True, alpha=0.3)

# ── Panel B: Experiment 2 vs Original (same σ=50, same t=0.08) ───────────────
ax = axes[1]
u_ana2 = np.array([u_exact(x_fixed, yv, t_exp2) for yv in y_fine])
ax.plot(y_fine, u_ana2, 'k-', lw=2, label=f'Analytical  t={t_exp2:.2f}s')

x_col2 = int(np.argmin(np.abs(xs2 - x_fixed)))
u_wos2 = u_exp2[:, x_col2]
ax.plot(ys2, u_wos2, 'o', color='steelblue', ms=7,
        label=f'WoS  σ=50  (20000 walks / step)')
ax.plot(ys0, u_wos0, 's', color='tomato', ms=6,
        label=f'WoS  σ=50  (2000 walks / step)  [original]')

ax.set_xlabel('y', fontsize=12)
ax.set_ylabel(f'u(x={x_fixed}, y, t)', fontsize=12)
ax.set_title('Experiment 2: Does increasing N_walks help?', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.grid(True, alpha=0.3)

fig.suptitle(
    'Diagnostic: root-cause of example04 WoS error\n'
    'Left: reduce σ (larger dt).  Right: increase N_walks (same σ=50).',
    fontsize=11)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), 'diagnostic_result.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved → {out}")

# ── Print summary errors ──────────────────────────────────────────────────────
print("\n── Error summary (MAE at 1D profile x=0.5) ──")
for label, ys_arr, u_wos, t_val in [
    ('Original  (σ=50, 2000 walks)', ys0, u_wos0, t_orig),
    ('Exp1      (σ=2,  2000 walks)', ys1, u_exp1[:, x_col1], t_exp1),
    ('Exp2      (σ=50, 20000 walks)', ys2, u_wos2, t_exp2),
]:
    u_ref = np.array([u_exact(x_fixed, yv, t_val) for yv in ys_arr])
    mae = np.mean(np.abs(u_wos - u_ref))
    print(f"  {label}  t={t_val:.2f}s  MAE={mae:.4f}")