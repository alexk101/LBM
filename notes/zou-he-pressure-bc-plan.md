# Faithful Zou–He Pressure Boundary Condition: Problem and Plan

## Problem statement

The LBM codebase in this project has a **pressure** (density) boundary condition that is currently implemented as an **approximation**, not the original Zou–He (1997) formulation. We want to replace it with a **faithful** implementation of the pressure BC as defined in:

**Zou, Q. and He, X. (1997). “On pressure and velocity boundary conditions for the lattice Boltzmann BGK model.” *Physics of Fluids*, 9(6), 1591–1598.**

### Current state

- **Velocity BC:** Already implements Zou–He correctly. Prescribed velocity **u** at the face; density **ρ** is derived from mass conservation; unknown populations are set via the non-equilibrium bounceback. The algebraic formulas (e.g. `f1 = f3 + (2/3)*ρ*u_x` for the west face) are the D2Q9 equivalent of the paper.
- **Pressure BC:** Currently uses a **simplified, stable approximation**: prescribed **ρ** at the face; velocity is taken from the **interior neighbor** (zero-gradient extrapolation); unknown populations are set to **full equilibrium** at (ρ, u_interior). There is **no** non-equilibrium bounceback. This avoids blow-up but is not the formulation in the paper and can show mass drift.

### Target state

At a pressure boundary we have:

- **Given:** Prescribed density **ρ** at the face.
- **Unknown:** The velocity **u** at the boundary and the distribution components that “stream from outside” (e.g. on an east outlet: f3, f6, f7).

In Zou–He (1997), boundary conditions are based on **bounceback of the non-equilibrium part** of the distribution:

- **f_i − f_i^eq(ρ,u) = f_opposite − f_opposite^eq(ρ,u)**

So the unknown population is:

- **f_i = f_i^eq(ρ,u) + (f_opposite − f_opposite^eq(ρ,u))**

The paper determines **u** from the requirement that mass and momentum are consistent with this rule. The task is to implement that: derive **u** from the momentum equations (with unknown f_i expressed by the formula above), then set the unknown f_i using the same non-equilibrium bounceback.

---

## Plan (implementation steps)

The following steps are enough for someone to implement the faithful Zou–He pressure BC without having to re-derive the method from the paper.

### 1. Location and reuse

- Implement the new logic in **`lbm/boundaries.py`**. No change to the solver’s public API (still `boundary_pressure={(axis, side): rho_face}` and optional `macro` if needed).
- Reuse:
  - **`_equilibrium_d2q9(rho, u, lattice)`** for f^eq(ρ,u).
  - **`_D2Q9_UNKNOWN[(axis, side)]`** for which directions are unknown on each face.
  - The lattice’s **opposite indices** (direction j opposite to direction i).

### 2. Express unknown populations in terms of u

For each pressure face (e.g. east):

- **Known:** The streamed f at the boundary node for directions that came from inside the domain (e.g. f0, f1, f2, f4, f5, f8).
- **Unknown:** f_i for directions that “stream from outside” (e.g. i ∈ {3, 6, 7} for east).

For each unknown direction **i** with opposite direction **j** (so that e_j = −e_i), set:

- **f_i = f_i^eq(ρ, u) + (f_j − f_j^eq(ρ, u))**

Here **f_j** is the **current** (streamed) value at the boundary for direction j. So every unknown f_i is a function of **(ρ, u)** and the known f_j. Implement this either symbolically or as a function that, given (ρ, u) and the streamed f at the node, returns the unknown f_i.

### 3. Close the system with momentum

The boundary (ρ, u) must satisfy:

- **ρ u_x = Σ_i f_i e_i,x**
- **ρ u_y = Σ_i f_i e_i,y**

Split the sums into:

- Contributions from **known** directions (use the current streamed f).
- Contributions from **unknown** directions (use the expression from step 2 in terms of u).

This yields two equations in (u_x, u_y). With the usual second-order equilibrium, the dependence on u is linear (plus quadratic terms), so you obtain either:

- A **2×2 linear system** in (u_x, u_y) if you keep only the linear-in-u part of f^eq in the unknown contributions, or  
- A **2×2 nonlinear system** if you keep the full quadratic equilibrium.

### 4. Solve for u

- **Option A (recommended first):** Derive the 2×2 **linear** system for (u_x, u_y) from the momentum equations (using the linear part of f^eq for the unknown directions). Solve it (e.g. `jnp.linalg.solve` or equivalent). Then compute all unknown f_i from the non-equilibrium bounceback formula using this u.
- **Option B:** Retain the full quadratic f^eq and solve the nonlinear 2×2 for u (e.g. a few Newton or fixed-point iterations), then compute the unknown f_i as above.

This solve is done per boundary node (or per face if the problem is 1D in the normal direction).

### 5. Write the boundary populations

Once u is known at the boundary:

- For each unknown direction i (with opposite j):  
  **f_i = f_i^eq(ρ, u) + (f_j − f_j^eq(ρ, u))**  
  using the **streamed** f_j at that node.
- Overwrite only those components (the unknown directions) in the boundary slice of the distribution array.

### 6. Robustness

- **Division by ρ:** When forming the linear system or computing u, guard against small ρ (e.g. `rho_safe = max(rho, eps)`).
- **Clamping (optional):** After solving for u, optionally clamp (u_x, u_y) to a reasonable range (e.g. ±0.3) to avoid rare instabilities.
- **Consistency check (optional):** In debug or tests, verify that with the new f_i, Σ f_i = ρ and Σ f_i e_i = ρ u at the boundary.

### 7. Validation

- **Poiseuille flow:** Run a channel with a pressure difference between inlet and outlet; compare the velocity profile to the analytical parabolic solution (as in Zou–He 1997).
- **Mass balance:** After a transient, total mass in the domain (or net flux in vs out) should stabilize; compare with the current equilibrium-based pressure BC.
- **Stability:** Run for long time and/or larger pressure gradients; ensure no NaNs or unbounded growth.

### 8. Documentation

- In **`lbm/boundaries.py`**, update the module and pressure-BC docstrings to state that the **pressure** BC now implements the Zou–He (1997) pressure condition: prescribed ρ, u from momentum consistency with non-equilibrium bounceback, and unknown f_i from the same bounceback.
- Leave the **velocity** BC description as is (already correct Zou–He).

---

## Summary checklist

| Step | Action |
|------|--------|
| 1 | Implement in `boundaries.py`; reuse equilibrium helper, unknown sets, and opposite indices. |
| 2 | For each pressure face, express unknown f_i = f_i^eq(ρ,u) + (f_opp − f_opp^eq(ρ,u)). |
| 3 | Write momentum equations ρ u = Σ f_i e_i with those f_i substituted; obtain 2×2 system in u. |
| 4 | Solve the 2×2 (linear or nonlinear) for u at the boundary. |
| 5 | Set f_i on the boundary from the non-equilibrium bounceback formula using that u. |
| 6 | Add safeguards (rho_safe, optional u clamping). |
| 7 | Validate with Poiseuille and mass balance; update docstrings. |

---

## Reference

- Zou, Q. and He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Physics of Fluids*, 9(6), 1591–1598.
