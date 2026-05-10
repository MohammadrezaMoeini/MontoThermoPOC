"""
Example 05: Transient heat equation on [0,1]² — Finite Element Method (P1 triangles)

Same problem as Example 04 (WoS), solved deterministically with FEM.

PDE:  ∂u/∂t = α ∇²u          on  Ω = [0,1]²
BC:   u = 1  (top,  y = 1),   u = 0  (bottom, left, right)
IC:   u(x, y, 0) = 0

Time discretisation: backward Euler
    M u^{n+1} + α Δt K u^{n+1} = M u^n
    ⟹  A u^{n+1} = b^n     where  A = M + α Δt K

Spatial discretisation: P1 (linear) triangular FEM on a uniform structured mesh.
Each unit square is split into two right triangles along the diagonal.

Dirichlet BCs are enforced by replacing BC rows with identity equations.
The system matrix is constant across time steps → LU-factored once for speed.

Run with:
    source ~/.MontoThermoPOC312/bin/activate
    python experimentDev/example05/example05.py
"""

import math
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ── Simulation parameters ─────────────────────────────────────────────────────
ALPHA   = 1.0          # thermal diffusivity
DT      = 0.02         # time step  (same as example04)
N_STEPS = 20           # t_final = 0.40
N_FEM   = 50           # n×n sub-squares → (n+1)² nodes, 2n² triangles

SAVE_AT = {0, 1, 2, 4, 8, 20}   # same save points as example04

# ── Analytical solution (identical to example04) ──────────────────────────────
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

# ── Mesh ──────────────────────────────────────────────────────────────────────
def build_mesh(n):
    """
    Uniform n×n mesh on [0,1]².
    Nodes: node(i,j) = i*(n+1)+j  with coordinates (i/n, j/n).
    Triangles: each square split along the SW–NE diagonal into two triangles.
    Returns x_coords, y_coords (length (n+1)²) and triangles (2n², 3).
    """
    h = 1.0 / n
    n_nodes = (n + 1) ** 2
    x_coords = np.empty(n_nodes)
    y_coords = np.empty(n_nodes)
    for i in range(n + 1):
        for j in range(n + 1):
            idx = i * (n + 1) + j
            x_coords[idx] = i * h
            y_coords[idx] = j * h

    tri_list = []
    for i in range(n):
        for j in range(n):
            bl = i * (n + 1) + j
            br = (i + 1) * (n + 1) + j
            tr = (i + 1) * (n + 1) + (j + 1)
            tl = i * (n + 1) + (j + 1)
            tri_list.append([bl, br, tr])   # lower triangle
            tri_list.append([bl, tr, tl])   # upper triangle

    return x_coords, y_coords, np.array(tri_list, dtype=np.int32)

# ── FEM assembly ──────────────────────────────────────────────────────────────
def assemble(x_coords, y_coords, triangles):
    """
    Assemble global stiffness K and consistent mass M matrices.

    Local stiffness:  K_ab = area * dot(∇φ_a, ∇φ_b)
    Local mass:       M_ab = area/12 * (1 + δ_ab)    (consistent mass matrix)
    """
    n_nodes = len(x_coords)
    n_tri   = len(triangles)

    # Pre-allocate COO arrays: 9 entries per triangle for each matrix
    rows = np.empty(9 * n_tri, dtype=np.int32)
    cols = np.empty(9 * n_tri, dtype=np.int32)
    kvals = np.empty(9 * n_tri)
    mvals = np.empty(9 * n_tri)

    ptr = 0
    for tri in triangles:
        a, b, c = tri
        xa, ya = x_coords[a], y_coords[a]
        xb, yb = x_coords[b], y_coords[b]
        xc, yc = x_coords[c], y_coords[c]

        # Signed area (triangles are CCW by construction)
        area = 0.5 * ((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))

        # Gradients of shape functions (constant on P1 element)
        inv2A = 0.5 / area
        grads = np.array([
            [(yb - yc) * inv2A, (xc - xb) * inv2A],
            [(yc - ya) * inv2A, (xa - xc) * inv2A],
            [(ya - yb) * inv2A, (xb - xa) * inv2A],
        ])

        nodes = [a, b, c]
        for ii in range(3):
            for jj in range(3):
                rows[ptr] = nodes[ii]
                cols[ptr] = nodes[jj]
                kvals[ptr] = area * np.dot(grads[ii], grads[jj])
                mvals[ptr] = area / 12.0 * (2.0 if ii == jj else 1.0)
                ptr += 1

    K = sp.csr_matrix((kvals, (rows, cols)), shape=(n_nodes, n_nodes))
    M = sp.csr_matrix((mvals, (rows, cols)), shape=(n_nodes, n_nodes))
    return K, M

# ── Dirichlet boundary conditions ─────────────────────────────────────────────
def get_dirichlet_nodes(x_coords, y_coords):
    """Return {node_index: value} for all Dirichlet boundary nodes."""
    tol = 1e-12
    bc = {}
    for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
        if abs(y - 1.0) < tol:
            bc[idx] = 1.0           # top edge: u = 1
        elif abs(y) < tol or abs(x) < tol or abs(x - 1.0) < tol:
            bc[idx] = 0.0           # other edges: u = 0
    return bc

def apply_dirichlet_rows(A, bc_indices):
    """
    Replace rows for Dirichlet nodes with identity rows.
    Columns are left intact so that the free-node equations automatically
    account for the BC contribution in the RHS via b = M @ u_prev.
    """
    A_lil = A.tolil()
    for idx in bc_indices:
        A_lil[idx, :] = 0.0
        A_lil[idx, idx] = 1.0
    return A_lil.tocsr()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Transient heat equation – FEM P1 triangles (2D)")
    print(f"  α = {ALPHA},   Δt = {DT},   {N_STEPS} steps,   t_final = {N_STEPS * DT:.2f}")
    n_nodes = (N_FEM + 1) ** 2
    n_tri   = 2 * N_FEM ** 2
    print(f"  Mesh: {N_FEM}×{N_FEM} squares → {n_tri} triangles,  {n_nodes} nodes")
    print("=" * 65)

    # ── Build mesh, matrices, BCs ─────────────────────────────────────────────
    x_coords, y_coords, triangles = build_mesh(N_FEM)
    K, M = assemble(x_coords, y_coords, triangles)
    bc = get_dirichlet_nodes(x_coords, y_coords)

    bc_indices = np.array(list(bc.keys()), dtype=np.int32)
    bc_values  = np.array([bc[i] for i in bc_indices])

    # System matrix A = M + α Δt K  (constant → factorise once)
    A = M + ALPHA * DT * K
    A_mod = apply_dirichlet_rows(A, bc_indices)
    A_lu  = spla.splu(A_mod)

    # ── Initial condition: u = 0 on interior, BC values on boundary ───────────
    u = np.zeros(n_nodes)
    u[bc_indices] = bc_values
    snapshots = {0: u.copy()}

    # ── Time loop ─────────────────────────────────────────────────────────────
    for step in range(1, N_STEPS + 1):
        b = M @ u                          # RHS from previous step
        b[bc_indices] = bc_values          # enforce Dirichlet in RHS
        u = A_lu.solve(b)
        if step in SAVE_AT:
            snapshots[step] = u.copy()
        print(f"  step {step:3d}/{N_STEPS}   t = {step * DT:.3f}")

    # ── Build evaluation grid for plotting ────────────────────────────────────
    # node(i,j) = i*(N_FEM+1)+j  → reshape to (N_FEM+1, N_FEM+1) grid [i,j]
    # For plotting: row=y, col=x → grid[j, i]
    nn = N_FEM + 1
    xi = np.linspace(0.0, 1.0, nn)        # x values (i-direction)
    yj = np.linspace(0.0, 1.0, nn)        # y values (j-direction)
    X, Y = np.meshgrid(xi, yj)            # shape (nn, nn), row=y, col=x

    def solution_to_grid(u_vec):
        """Reshape flat node vector to (nn, nn) grid [j, i]."""
        grid = np.empty((nn, nn))
        for i in range(nn):
            for j in range(nn):
                grid[j, i] = u_vec[i * nn + j]
        return grid

    # ── Error at final step ───────────────────────────────────────────────────
    t_final = N_STEPS * DT
    Z_fem_final = solution_to_grid(snapshots[N_STEPS])
    Z_ana_final = np.vectorize(u_exact)(X, Y, t_final)
    err = np.abs(Z_fem_final - Z_ana_final)
    print(f"\nFinal (t={t_final:.2f}) vs analytical:  "
          f"max_err = {err.max():.5f},  mean_err = {err.mean():.5f}")

    # ── 2D contour plots: FEM (top) vs analytical (bottom) ───────────────────
    steps_to_plot = sorted(snapshots.keys())
    ncols = len(steps_to_plot)

    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))

    for col, step in enumerate(steps_to_plot):
        t_val = step * DT
        Z_fem = solution_to_grid(snapshots[step])
        Z_ana = np.vectorize(u_exact)(X, Y, t_val)

        vmin   = float(min(Z_fem.min(), Z_ana.min()))
        vmax   = float(max(Z_fem.max(), Z_ana.max()))
        levels = np.linspace(vmin, vmax, 21)

        im = axes[0, col].contourf(X, Y, Z_fem, levels=levels, cmap="turbo")
        plt.colorbar(im, ax=axes[0, col])
        axes[0, col].set_title(f"FEM   t = {t_val:.2f}", fontsize=9)
        axes[0, col].set_xlabel("x"); axes[0, col].set_ylabel("y")

        im = axes[1, col].contourf(X, Y, Z_ana, levels=levels, cmap="turbo")
        plt.colorbar(im, ax=axes[1, col])
        axes[1, col].set_title(f"Analytical   t = {t_val:.2f}", fontsize=9)
        axes[1, col].set_xlabel("x"); axes[1, col].set_ylabel("y")

    fig.suptitle(
        f"Transient heat  ∂u/∂t = α∇²u  on [0,1]²   |   u=1 top, u=0 elsewhere\n"
        f"α = {ALPHA},   Δt = {DT},   FEM P1 ({N_FEM}×{N_FEM} mesh)\n"
        f"Top: FEM  |  Bottom: exact Fourier series",
        fontsize=11, y=1.01)
    plt.tight_layout()

    out_2d = os.path.join(os.path.dirname(__file__), "fem_solution.png")
    plt.savefig(out_2d, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_2d}")

    # ── 1D profile: u vs y at x=0.5, one subplot per saved time step ─────────
    x_fixed = 0.5
    # Find the FEM nodes at x ≈ 0.5 (i-index closest to x=0.5)
    i_fixed = int(round(x_fixed * N_FEM))
    x_actual = i_fixed / N_FEM
    j_indices = np.arange(nn)
    y_nodes   = j_indices / N_FEM          # y values at those nodes

    y_fine = np.linspace(0.0, 1.0, 300)

    ncols1d = len(steps_to_plot)
    fig1d, axes1d = plt.subplots(1, ncols1d, figsize=(4 * ncols1d, 4), sharey=True)

    for ax, step in zip(axes1d, steps_to_plot):
        t_val = step * DT
        # FEM values along the vertical line x = x_actual
        u_fem = np.array([snapshots[step][i_fixed * nn + j] for j in j_indices])
        u_ana = np.array([u_exact(x_actual, yv, t_val) for yv in y_fine])

        ax.plot(y_fine, u_ana, color="steelblue", lw=2, label="Analytical")
        ax.plot(y_nodes, u_fem, color="tomato", lw=1.5, marker="o",
                ms=3, label="FEM")

        ax.set_title(f"t = {t_val:.2f}", fontsize=10)
        ax.set_xlabel("y", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes1d[0].set_ylabel(f"u(x={x_actual:.3f},  y,  t)", fontsize=10)
    fig1d.suptitle(
        f"FEM 1D profile at x = {x_actual:.3f}  —  solid: analytical,  line+dots: FEM",
        fontsize=11)
    plt.tight_layout()

    out_1d = os.path.join(os.path.dirname(__file__), "fem_1d_profile.png")
    plt.savefig(out_1d, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_1d}")


main()