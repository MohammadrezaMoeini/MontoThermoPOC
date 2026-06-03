# Example 07 — Validation Against Trofimov 2022

## Goal

Validate the `TransientThermalSolver` + `GCodeTransientSimulator` against the
experimental thermographic data in:

> Trofimov, A. et al. (2022). *Experimentally validated modeling of temperature
> distribution during FFF.* `tests/Anton/2022_Trofimov_...pdf`

The paper provides 27 IR-measured T vs time curves at known coordinates on a
printed PLA plate. We run our solver on the same geometry and compare.

### Scope: geometry (a), layer 1 only
The paper validates two geometries:
- **(a) Regular plate** — 60 mm × 10 mm × 4 mm rectangular solid. **This is our target.**
- **(b) Bridge-like structure** — out of scope for this example.

Files for geometry (a): `experimentDev/example07_ValidationWithAnton/geometry.stl`,
`experimentDev/example07_ValidationWithAnton/geometry.gcode`

**Active simulation scope**: points **#1–9 only** (layer 1, z = 0.2 mm in paper /
z = 0.4 mm in the GCode). This is the first and most challenging layer (highest
thermal gradients, bed contact effects). Layers 10 and 20 (points 10–27) are kept
in `validation.py` for future use but are not simulated in `example07.py`.

---

## Paper Key Facts (Trofimov 2022)

### Geometry
- 60 mm × 10 mm × 4 mm PLA rectangular plate
- Paper: 20 layers × 0.2 mm layer height
- **GCode file (actual)**: 10 layers × 0.4 mm layer height (same total height = 4 mm)
- GCode coordinate system: centred at plate centre (x ∈ [−4.4, 4.4] mm, y ∈ [−29.52, 29.52] mm)
- Coordinate transform → paper-to-GCode: `x_g = x_p − 4.4`, `y_g = y_p − 29.52`
- GCode move structure: 230 long beads (≈59 mm, E Δ≈3.93) + 220 lateral x-steps (0.4 mm, E Δ≈0.027)
  → filter x-steps with `length < 1 mm` before simulation

### Printer / Process settings
| Parameter | Value |
|---|---|
| Nozzle set temp | 210 °C |
| Bed set temp | 60 °C |
| Nozzle diameter | 0.4 mm |
| Layer height | 0.2 mm |
| Printer | Raise3D Pro2 |

### Measured / calibrated parameters (Table 2)
| Symbol | Meaning | Value |
|---|---|---|
| T_b | Bed temperature (measured) | 52 °C |
| T_s | Deposition surface temp (measured) | 60 °C |
| h_f | Free-surface convection coefficient | 71 W/(m²·K) |
| h_b | Bed-contact convection coefficient | 5 W/(m²·K) |
| k | PLA thermal conductivity | 0.11 W/(m·K) |
| ρ | PLA density | 1250 kg/m³ |
| cp | PLA specific heat | 1590 J/(kg·K) |
| Δt | Time step used in paper's FEM | 0.01 s |

### Validation points (Figure 10)
27 points = 3 layers × 9 (x, y) positions.

Layer z-values: **layer 1** → z ≈ 0.2 mm, **layer 10** → z ≈ 2.0 mm, **layer 20** → z ≈ 4.0 mm

| Point group | x [mm] | y [mm] |
|---|---|---|
| 1, 2, 3 | 0.4 | 0.4, 29.6, 44.0 |
| 4, 5, 6 | 4.0 | 0.4, 29.6, 44.0 |
| 7, 8, 9 | 8.8 | 0.4, 29.6, 44.0 |

(x = depth into width = 0…10 mm; y = position along length = 0…60 mm)

Paper accuracy: 25/27 points < 1 °C error; max discrepancy ≈ 24 °C at
points 4 & 7 (very close to bed — strong h_b gradient).

---

## Validation Strategy (agreed with user)

### Step 1 — Parameter alignment
Instantiate `TransientThermalSolver` with the paper's measured values:
```python
solver = TransientThermalSolver(
    T_bed     = 52.0,    # T_b measured (Table 2)
    T_nozzle  = 210.0,   # nozzle set temperature (Section 3, p.6)
    T_ambient = 20.0,
    h         = 71.0,    # h_f free-surface convection (Table 2)
    k_cond    = 0.11,
    rho       = 1250.0,
    cp        = 1590.0,
    dt        = 0.5,
)
```

Note: T_s = 60°C in Table 2 is the measured temperature of the previously deposited
layer surface — it is NOT the nozzle BC. The nozzle Dirichlet BC is 210°C.

### Step 2 — Run the Anton GCode
Feed `tests/Anton/geometry.gcode` into `GCodeTransientSimulator`.
Parse and inspect the file first to confirm layer height, feed rate, and
total move count.

### Step 3 — Extract T at the 9 layer-1 validation coordinates
After simulation, for each of points #1–9 (layer 1), find the nearest
simulation grid point and extract the full T vs time history.
Plot overlaid on subplots (a), (b), (c) of `validation.py` (Exp + FEM already there).

Layer-1 point coordinates in GCode space (transform: x_g = x_p − 4.4, y_g = y_p − 29.52):
| Paper pt | x_paper | y_paper | x_gcode | y_gcode | z_gcode |
|---|---|---|---|---|---|
| 1 | 0.4 | 0.4  | −4.0 | −29.12 | 0.4 |
| 2 | 0.4 | 29.6 | −4.0 |   0.08 | 0.4 |
| 3 | 0.4 | 44.0 | −4.0 |  14.48 | 0.4 |
| 4 | 4.0 | 0.4  | −0.4 | −29.12 | 0.4 |
| 5 | 4.0 | 29.6 | −0.4 |   0.08 | 0.4 |
| 6 | 4.0 | 44.0 | −0.4 |  14.48 | 0.4 |
| 7 | 8.8 | 0.4  |  4.4 | −29.12 | 0.4 |
| 8 | 8.8 | 29.6 |  4.4 |   0.08 | 0.4 |
| 9 | 8.8 | 44.0 |  4.4 |  14.48 | 0.4 |

### Step 4 — Quantitative comparison (layer 1 only)
Compute RMS error and max error over points #1–9.
Points #4 and #7 are expected to show the largest discrepancy (~24 °C) because
they are in direct contact with the bed (h_b effect not yet modelled separately).
Points #10–27 (layers 10 & 20) remain in `validation.py` for future extension.

---

## Known Gaps vs the Paper's Model

| # | Gap | Impact | Fix needed? |
|---|---|---|---|
| 1 | Paper nozzle BC = 210°C; T_s=60°C is layer surface temp (not nozzle BC) | **Resolved** — config now uses T_nozzle=210°C | Done |
| 2 | Single h for all surfaces; paper separates h_f / h_b | **Medium** — affects bed-contact points 4&7 | Add bottom-face Robin BC with h_b=5 |
| 3 | No radiation BC | Low at these temps | Ignore for now |
| 4 | Constant material properties | Low | Ignore for now |
| 5 | Mean T_ic carried between beads (not full plate FEM) | **High** — biggest structural difference | Known limitation; note in results |

---

## Current Solver Architecture (as of example 06 / June 2026)

```
src/gcode/parser.py          GCodeParser — parses .gcode → list of move dicts
src/gcode/geometry.py        GeometryBuilder — single-bead cuboid mesh
solver/thermal_solver.py     ThermalSolver (base) — steady-state WoS
solver/transient_thermal_solver.py  TransientThermalSolver — backward Euler + WoS
solver/gcode_transient_simulator.py GCodeTransientSimulator — bead-by-bead loop
experimentDev/example06/     T vs time at 3 z-positions — reference example
tests/test_transient_solver.py
tests/test_gcode_transient_simulator.py
```

### Key solver equations
- PDE per time step: `∇²T^(n+1) − σ·T^(n+1) = −σ·T^n`
- σ = 1/(α·Δt),  α = k/(ρ·cp) × 10⁶  [mm²/s]
- zombie API: `pde.absorption_coeff = σ`, source via `get_dense_grid_source_callback`

### Result dict keys (per bead)
`bead_idx, move, dt_deposition, t_start, t_end, T_mid_history, T_mid_times,
T_final_mean, points, T_deposition, T_cooling`

---

## Files to Create in This Folder

- `example07.py` — main validation script
- `SUMMARY.md` — this file
- `T_validation_*.png` — output comparison plots (generated, not committed)