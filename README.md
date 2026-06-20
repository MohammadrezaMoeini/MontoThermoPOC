# MontoThermoPOC: 
This is a thermal simulator by MC. 

# Examples
In experimentDev folder, there are verification and validation examples. 

## Example 01:
This example is a first draft of Monte Carlo simulation from scratch without using any library.


## Example 02:
In this example, a 2D heat problem was solved with zombie library. 
### PDE & B.C.
The equation that we solved is laplace equation and with Dirichlet boundary conditions: 

$ \nabla^2 u = 0$ on $[0,1]^2$
                                                                                                                                                                                                                                         
  **Boundary conditions (Dirichlet):**                                                                                                                                                                                                   
                                                                                                                                                                                                                                         
  | Edge   | $(x, y)$ | Condition |                                                                                                                                                                                                            
  |--------|----------|-----------|
  | Bottom | $y=0$    | $u = 0$   |
  | Left   | $x=0$    | $u = 0$   |
  | Right  | $x=1$    | $u = 0$   |
  | Top    | $y=1$    | $u = 1$   |

### Output
The output would be the predicted temprature inside the target domain of [0, 1]

![Example 2](experimentDev/example02/laplace_solution.png)
* Example 2: Solving laplace equation using zombie library and comparing the analytical solution


## Example 03:
In this example, the 3D Laplace equation was solved on the unit cube $[0,1]^3$ using the zombie Walk-on-Stars library.

### PDE & B.C.

$$
\nabla^2 u = 0 \quad \text{on } [0,1]^3
$$

**Boundary conditions (Dirichlet):**

| Face        | Condition |
|-------------|-----------|
| Top ($z=1$) | $u = 1$   |
| All others  | $u = 0$   |

### Output
The solution is visualised at three horizontal slices ($z = 0.25$, $0.50$, $0.75$), each compared against the analytical double Fourier-series solution.

![Example 3 — 2D slices](experimentDev/example03/laplace_3d_solution.png)
* Example 3: Analytical vs Zombie WoS solution and absolute error at three $z$-slices

![Example 3 — 3D view](experimentDev/example03/laplace_3d_plot.png)
* Example 3: 3D view of the solution at horizontal slices coloured by temperature

![Example 3 — 1D cross-section](experimentDev/example03/laplace_1d_crosssection.png)
* Example 3: 1D cross-section along $y=0.5$, $z=0.5$ — Analytical vs Zombie WoS


## Example 04: 

In this example, the transient heat equation was solved on $[0,1]^2$ using the zombie Walk-on-Stars library with a backward Euler time discretisation.

### PDE & B.C.

$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u \quad \text{on } [0,1]^2
$$

**Boundary conditions (Dirichlet):**

| Edge          | Condition |
|---------------|-----------|
| Top ($y=1$)   | $u = 1$   |
| All others    | $u = 0$   |

**Initial condition:** $u(x, y, 0) = 0$

### Time discretisation

Backward Euler converts the PDE into a screened-Poisson (Yukawa) equation at each step:

$$
\nabla^2 u^{n+1} - \sigma\, u^{n+1} = -\sigma\, u^n \qquad \sigma = \frac{1}{\alpha \Delta t}
$$

Zombie solves this directly with $\lambda = \sigma$ and source $f = \sigma \cdot u^n$.
At steady state ($u^{n+1} \approx u^n$) the equation collapses back to $\nabla^2 u = 0$, recovering the Example 02 Laplace solution.

### Key constraint: $\sigma$ must stay small

The Dirichlet boundary influence on interior points decays as $\exp(-\sqrt{\sigma} \cdot d)$. For large $\sigma$ this signal becomes undetectable, causing the solution to stagnate. Two diagnostic experiments confirmed this:

- **Increasing N\_walks 10×** (2 000 → 20 000) with $\sigma = 50$: error unchanged — not a variance issue
- **Reducing $\sigma$** from 50 to 2 ($\Delta t$: 0.02 → 0.5): mean error dropped 9× — confirmed root cause

The working parameters use $\sigma = 2$:

| Parameter | Value |
|-----------|-------|
| dt | 0.5 s |
| sigma | 2 |
| N\_steps | 6 (t\_final = 3.0 s) |
| N\_walks | 2 000 |
| Mean error | 0.017 |

### Output
Snapshots at $t = 0,\, 0.5,\, 1.0,\, 1.5,\, 2.0,\, 3.0$ compared against the analytical Fourier-series solution.

![Example 4 — 2D snapshots](experimentDev/example04/transient_solution.png)
* Example 4: WoS (top) vs analytical (bottom) temperature field at six time snapshots

![Example 4 — 1D profiles](experimentDev/example04/transient_1d_profile.png)
* Example 4: 1D profile at $x=0.5$ — analytical (line) vs WoS (dots) at each saved time


## Example 05: 
This example solves the same transient heat problem as Example 04, but deterministically using the **Finite Element Method** (P1 linear triangles) instead of Walk-on-Stars. It serves as a reference solver to benchmark the WoS results.

### PDE & B.C.

$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u \quad \text{on } [0,1]^2
$$

**Boundary conditions (Dirichlet):**

| Edge          | Condition |
|---------------|-----------|
| Top ($y=1$)   | $u = 1$   |
| All others    | $u = 0$   |

**Initial condition:** $u(x, y, 0) = 0$

### Time discretisation

Backward Euler with the FEM system matrix $A = M + \alpha \Delta t\, K$ (factored once via LU):

$$
A\, u^{n+1} = M\, u^n
$$

where $M$ is the consistent mass matrix and $K$ is the stiffness matrix assembled from P1 shape functions on a uniform $50 \times 50$ triangle mesh.

### Output

![Example 5 — 2D snapshots](experimentDev/example05/fem_solution.png)
* Example 5: FEM (top) vs analytical (bottom) temperature field at six time snapshots

![Example 5 — 1D profiles](experimentDev/example05/fem_1d_profile.png)
* Example 5: 1D profile at $x=0.5$ — analytical (line) vs FEM (dots) at each saved time

![Example 5 — WoS vs FEM comparison](experimentDev/example05/wos_vs_fem.png)
* Example 5: WoS (example04) vs FEM vs Analytical — 1D profile at $x=0.5$, backward Euler $\Delta t=0.5$, $\sigma=2$







