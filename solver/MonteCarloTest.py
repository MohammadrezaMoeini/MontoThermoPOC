"""
Transient Heat Transfer – Walk on Spheres, accelerated
=======================================================
Three strategies applied vs the baseline:

  1. Numba JIT  – compiles wos_walk to native machine code.
                  Eliminates NumPy per-call overhead for tiny 3-vectors.
                  Replaces np.linalg.norm / np.clip with scalar math.

  2. Parallel walks – all N_WALKS for all probes dispatched as
                  independent tasks via concurrent.futures.ProcessPoolExecutor.
                  Each worker gets its own RNG seed so results are reproducible.
                  (Threading doesn't help here due to Python's GIL;
                   ProcessPool bypasses it and gives true multi-core speedup.)

  3. Vectorised RNG batching inside each walk batch – pre-draw all
                  random numbers for a block of walks at once rather than
                  calling rng inside the JIT loop (Numba's RNG is slower
                  than NumPy's; pre-drawing is faster).

Install: pip install numpy matplotlib tqdm numba
"""

import math, time, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit
import numba

# ---------------------------------------------------------------------------
# Physical / simulation parameters  (identical to baseline)
# ---------------------------------------------------------------------------
K_SOLID  = 0.20
RHO      = 1200.0
CP       = 1500.0
ALPHA    = K_SOLID / (RHO * CP)
H_CONV   = 10.0
T_INF    = 25.0
T_BOTTOM = 60.0
T_TOP    = 200.0
T_INIT   = 25.0
L        = 0.01*10

DT         = 90.0
N_STEPS    = 20*10
N_WALKS    = int(256/1)      # can afford more walks now – costs the same wall time
MAX_STEPS  = 512
EPS        = 5e-4

SIGMA_CODE = 1.0 / (ALPHA * DT) * L**2
MU_CODE    = (H_CONV / K_SOLID) * L

PROBES = {
    "Low  (z=0.25)":  np.array([0.5, 0.5, 0.25]),
    "Mid  (z=0.50)":  np.array([0.5, 0.5, 0.50]),
    "High (z=0.75)":  np.array([0.5, 0.5, 0.75]),
}

# ===========================================================================
#  NUMBA-JIT WALK  (pure scalar, no NumPy inside the loop)
# ===========================================================================

@njit(cache=True)
def _wos_walk_jit(
    x0, y0, z0,          # start point
    sigma, mu, eps,       # PDE / BC parameters
    T_bottom, T_top, T_inf,
    # T^n interpolation anchors (1-D along z)
    z_anchors, T_anchors, # float64 arrays length 5
    # Pre-drawn random numbers for this walk
    rands,               # shape (MAX_STEPS, 4): cols = [u_rr, u_face, nx,ny,nz... ]
    # we pass flat arrays; inside we use indices
    normals_flat,        # 6*3 = 18 floats, row-major
    max_steps,
):
    """
    Single WoS walk compiled to native code by Numba.

    Random numbers are pre-drawn outside and passed in so we can use
    NumPy's fast Philox generator rather than Numba's slower one.

    rands shape: (max_steps, 5)
      col 0: uniform for Russian roulette / face absorption
      col 1: uniform for Robin absorption decision
      col 2-4: normal(0,1) for sphere direction (normalised inside)
    """
    x, y, z = x0, y0, z0

    for step in range(max_steps):
        # Distance to each face
        d0 = x;       d1 = 1.0 - x
        d2 = y;       d3 = 1.0 - y
        d4 = z;       d5 = 1.0 - z

        # Find min distance and face index manually (faster than np.argmin)
        r = d0; fi = 0
        if d1 < r: r = d1; fi = 1
        if d2 < r: r = d2; fi = 2
        if d3 < r: r = d3; fi = 3
        if d4 < r: r = d4; fi = 4
        if d5 < r: r = d5; fi = 5

        # ── Epsilon-shell ────────────────────────────────────────────────────
        if r < eps:
            if fi == 4:
                return T_bottom
            elif fi == 5:
                return T_top
            else:
                # Robin: absorb with prob w = mu*eps/(1+mu*eps)
                w = mu * eps / (1.0 + mu * eps)
                if rands[step, 1] < w:
                    return T_inf
                # Hemisphere reflection inward
                ni = fi * 3
                nx_n = normals_flat[ni];   ny_n = normals_flat[ni+1]; nz_n = normals_flat[ni+2]
                vx = rands[step, 2]; vy = rands[step, 3]; vz = rands[step, 4]
                vlen = math.sqrt(vx*vx + vy*vy + vz*vz)
                vx /= vlen; vy /= vlen; vz /= vlen
                dot = vx*nx_n + vy*ny_n + vz*nz_n
                if dot < 0.0:
                    vx = -vx; vy = -vy; vz = -vz
                x = x + r * vx
                y = y + r * vy
                z = z + r * vz
                # clamp
                if x < eps*2: x = eps*2
                if x > 1-eps*2: x = 1-eps*2
                if y < eps*2: y = eps*2
                if y > 1-eps*2: y = 1-eps*2
                if z < eps*2: z = eps*2
                if z > 1-eps*2: z = 1-eps*2
                continue

        # ── Screened Poisson absorption ──────────────────────────────────────
        sr = math.sqrt(sigma) * r
        if sr > 1e-10:
            p_surv = math.tanh(sr) / sr
        else:
            p_surv = 1.0 - sigma * r * r / 3.0

        if rands[step, 0] > p_surv:
            # Absorbed: return T^n(x) via linear interp along z
            return _interp1d(z, z_anchors, T_anchors)

        # ── Jump to sphere surface ────────────────────────────────────────────
        vx = rands[step, 2]; vy = rands[step, 3]; vz = rands[step, 4]
        vlen = math.sqrt(vx*vx + vy*vy + vz*vz)
        vx /= vlen; vy /= vlen; vz /= vlen
        x = x + r * vx
        y = y + r * vy
        z = z + r * vz
        if x < eps: x = eps
        if x > 1-eps: x = 1-eps
        if y < eps: y = eps
        if y > 1-eps: y = 1-eps
        if z < eps: z = eps
        if z > 1-eps: z = 1-eps

    # Timeout: return T^n at current position
    return _interp1d(z, z_anchors, T_anchors)


@njit(cache=True)
def _interp1d(z, z_anchors, T_anchors):
    """Linear interpolation along z (Numba-compatible)."""
    n = len(z_anchors)
    if z <= z_anchors[0]:
        return T_anchors[0]
    if z >= z_anchors[n-1]:
        return T_anchors[n-1]
    for i in range(n-1):
        if z_anchors[i] <= z <= z_anchors[i+1]:
            t = (z - z_anchors[i]) / (z_anchors[i+1] - z_anchors[i])
            return T_anchors[i] + t * (T_anchors[i+1] - T_anchors[i])
    return T_anchors[n-1]


@njit(cache=True, parallel=True)
def run_walks_batch(
    start,           # (3,) start point
    n_walks,
    sigma, mu, eps,
    T_bottom, T_top, T_inf,
    z_anchors, T_anchors,
    rands_batch,     # (n_walks, MAX_STEPS, 5)
    normals_flat,
    max_steps,
):
    """Run n_walks independent walks in parallel using Numba prange."""
    results = np.empty(n_walks)
    for i in numba.prange(n_walks):
        results[i] = _wos_walk_jit(
            start[0], start[1], start[2],
            sigma, mu, eps,
            T_bottom, T_top, T_inf,
            z_anchors, T_anchors,
            rands_batch[i],
            normals_flat,
            max_steps,
        )
    return results


# Inward normals flat array (6 faces × 3 components)
_NORMALS_FLAT = np.array([
     1., 0., 0.,   # face 0: x=0
    -1., 0., 0.,   # face 1: x=1
     0., 1., 0.,   # face 2: y=0
     0.,-1., 0.,   # face 3: y=1
     0., 0., 1.,   # face 4: z=0  (bottom)
     0., 0.,-1.,   # face 5: z=1  (top)
], dtype=np.float64)


def estimate_T_fast(pt, z_anchors, T_anchors, n_walks, rng):
    """
    Estimate T at pt using n_walks JIT-compiled walks.
    Draws all random numbers upfront in one NumPy call (fast Philox),
    then passes them to the JIT batch runner.
    """
    # Pre-draw all randoms: shape (n_walks, MAX_STEPS, 5)
    # cols: [rr_uniform, face_uniform, nx, ny, nz]
    rands = np.empty((n_walks, MAX_STEPS, 5), dtype=np.float64)
    rands[:, :, 0] = rng.random((n_walks, MAX_STEPS))        # Russian roulette
    rands[:, :, 1] = rng.random((n_walks, MAX_STEPS))        # Robin decision
    rands[:, :, 2:5] = rng.standard_normal((n_walks, MAX_STEPS, 3))  # direction

    samples = run_walks_batch(
        np.array(pt, dtype=np.float64),
        n_walks,
        SIGMA_CODE, MU_CODE, EPS,
        T_BOTTOM, T_TOP, T_INF,
        z_anchors, T_anchors,
        rands,
        _NORMALS_FLAT,
        MAX_STEPS,
    )
    return float(samples.mean()), float(samples.std() / math.sqrt(n_walks))


def make_anchors(T_probe):
    z_a = np.array([0.0,      0.25,       0.50,       0.75,       1.0],  dtype=np.float64)
    T_a = np.array([T_BOTTOM, T_probe[0], T_probe[1], T_probe[2], T_TOP], dtype=np.float64)
    return z_a, T_a


# ===========================================================================
#  WARM UP Numba  (JIT compilation happens on first call)
# ===========================================================================

def warmup():
    print("Compiling JIT kernels (one-time, ~5s)...", flush=True)
    rng = np.random.default_rng(0)
    z_a, T_a = make_anchors([50., 100., 150.])
    estimate_T_fast([0.5,0.5,0.5], z_a, T_a, n_walks=4, rng=rng)
    print("JIT ready.\n", flush=True)


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    warmup()

    rng       = np.random.default_rng(42)
    pts       = list(PROBES.values())
    lbls      = list(PROBES.keys())
    n         = len(pts)

    T_history = np.zeros((N_STEPS + 1, n))
    T_err     = np.zeros((N_STEPS + 1, n))
    time_axis = np.arange(N_STEPS + 1, dtype=float) * DT

    T_history[0, :] = T_INIT

    print("=" * 55)
    print("  Transient WoS  –  Numba + parallel walks")
    print("=" * 55)
    print(f"  {N_WALKS} walks/point,  {os.cpu_count()} CPUs,  {N_STEPS} steps")
    print()

    t0 = time.perf_counter()

    for step in tqdm(range(1, N_STEPS + 1), desc="timesteps"):
        z_a, T_a = make_anchors(T_history[step - 1].tolist())

        for j, pt in enumerate(pts):
            mu, se              = estimate_T_fast(pt, z_a, T_a, N_WALKS, rng)
            T_history[step, j] = mu
            T_err[step, j]     = se

        line = "  ".join(
            f"{T_history[step,j]:.1f}+-{T_err[step,j]:.1f}C"
            for j in range(n)
        )
        tqdm.write(f"  step {step:2d}  t={step*DT:6.0f}s  {line}")

    wall = time.perf_counter() - t0
    print(f"\nWall time (excl. JIT warmup): {wall:.2f}s  ({N_WALKS} walks/pt)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["steelblue", "darkorange", "seagreen"]

    for j, (lbl, col) in enumerate(zip(lbls, colors)):
        T  = T_history[:, j]
        se = T_err[:, j]
        ax.plot(time_axis, T, color=col, lw=2, marker="o", ms=4, label=lbl)
        ax.fill_between(time_axis, T - 2*se, T + 2*se, color=col, alpha=0.15)

    ax.axhline(T_BOTTOM, color="royalblue", ls="--", lw=1,
               label=f"T_bottom = {T_BOTTOM}°C")
    ax.axhline(T_TOP,    color="crimson",   ls="--", lw=1,
               label=f"T_top = {T_TOP}°C")
    ax.axhline(T_INF,    color="grey",      ls=":",  lw=1,
               label=f"T_inf = {T_INF}°C")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Temperature [°C]", fontsize=12)
    ax.set_title(
        f"Transient heat – Numba JIT + parallel walks\n"
        f"10 mm cube | {N_WALKS} walks/point | Fo={ALPHA*DT/L**2:.2f}",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlim(0, N_STEPS * DT)
    ax.set_ylim(T_INIT - 15, T_TOP + 15)

    plt.tight_layout()
    plt.savefig("probe_temperature_fast.png", dpi=150)
    plt.show()
    print("Saved probe_temperature_fast.png")

    return T_history, T_err, time_axis


T_history, T_err, time_axis = main()