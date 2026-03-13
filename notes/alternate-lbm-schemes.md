# Alternate LBM schemes: polynomial, entropic, Levermore, extended, MRT, TRT, central-moment, regularized

This note explains different ways to define the **equilibrium distribution** and/or the **collision** step in the lattice Boltzmann method. Sections 1-4 focus on **equilibrium** choices; sections 5-8 on **collision** or **formulation** variants (same polynomial equilibrium, different relaxation or space). Each choice has different trade-offs in accuracy, stability, and range of applicability (e.g. incompressible vs compressible, low vs high Mach).

---

## 1. Polynomial equilibrium (what we have)

**Idea:** The continuous Maxwell-Boltzmann distribution is expanded in small Mach number and evaluated on the discrete velocities via the lattice weights (Gauss-Hermite quadrature). The result is a **polynomial in u**.

For D2Q9 with sound speed squared $c_s^2 = 1/3$, the standard form is:

$$
f_i^{\mathrm{eq}} = \rho\, w_i \left(
  1 + \frac{\mathbf{e}_i\cdot\mathbf{u}}{c_s^2}
  + \frac{(\mathbf{e}_i\cdot\mathbf{u})^2}{2 c_s^4}
  - \frac{|\mathbf{u}|^2}{2 c_s^2}
\right)
$$

**Pros:**

- Simple, explicit, cheap.
- Correct low-Mach Navier-Stokes limit (mass, momentum, stress) when used with BGK and appropriate $\tau$.
- Well understood and widely used.

**Cons:**

- Only accurate for small $|\mathbf{u}|/c_s$ (low Mach). At higher Mach, the truncation error grows.
- $f_i^{\mathrm{eq}}$ can become **negative** for large u; then $f_i$ can go negative and the scheme can blow up.
- No built-in guarantee that a discrete entropy (e.g. $H = \sum f_i \ln(f_i/w_i)$) never increases; stability is conditional (e.g. $\tau > 0.5$, small enough u).

**In this codebase:** `ParticleDistribution.equilibrium` in `lbm/distributions.py` implements this second-order polynomial equilibrium.

**Citation:** Bhatnagar, P.L., Gross, E.P., Krook, M. (1954). A model for collision processes in gases. I. Small amplitude processes in charged and neutral one-component systems. *Physical Review* **94**, 511-525. DOI: 10.1103/PhysRev.94.511. (BGK collision.) For the lattice Boltzmann formulation with this equilibrium: Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G., Viggen, E.M. (2017). *The Lattice Boltzmann Method*. Springer. Chapter 2.

---

## 2. Entropic LBM (ELBM)

**Idea:** Define a **discrete entropy** $H(\mathbf{f})$ (e.g. $H = \sum_i f_i \ln(f_i/w_i)$). The discrete equilibrium is the one that **minimizes** $H$ under the constraints of given density and momentum (and possibly energy in thermal models). Collision is then designed so that $H$ does not increase (discrete H-theorem).

So:

- Equilibrium: $\mathbf{f}^{\mathrm{eq}} = \mathrm{argmin}_{\mathbf{f}} H(\mathbf{f})$ subject to $\sum f_i = \rho$, $\sum f_i \mathbf{e}_i = \rho\mathbf{u}$ (and optionally energy).
- Collision: often an “entropy fix” (e.g. adaptive $\alpha$ in $f \to f + \alpha\,\omega\,(f^{\mathrm{eq}} - f)$) so that the post-collision state still satisfies the entropy condition.

**Pros:**

- Equilibrium is **positive** by construction (minimum of entropy under linear constraints).
- With an entropy-stable collision, the scheme can have much better **stability** at high Mach and high Reynolds number.
- Theoretically clean (connection to kinetic theory and H-theorem).

**Cons:**

- Equilibrium is **implicit**: solve a convex minimization for $\mathbf{f}^{\mathrm{eq}}(\rho,\mathbf{u})$ (and possibly $T$) at each node; more cost per node than a polynomial.
- Tuning the entropy fix (e.g. $\alpha$) can affect dissipation (e.g. bulk viscosity, acoustic damping).
- Implementation and analysis are more involved than polynomial BGK.

**Typical use:** Compressible flows, high Mach, stability-sensitive setups; also used in athermal and thermal LBMs when robustness is prioritized.

**Citations:** Ansumali, S., Karlin, I.V., Öttinger, H.C. (2003). Minimal entropic kinetic models for hydrodynamics. *Europhysics Letters* **63**, 798-804. Karlin, I.V., Ansumali, S., De Angelis, E., Öttinger, H.C., Succi, S. (2003). Entropic lattice Boltzmann method for large scale turbulence simulation. *International Journal of Modern Physics C* **14**, 1111-1123.

---

## 3. Levermore (moment-closure) approach

**Idea:** In kinetic theory, **Levermore** (and collaborators) built **moment-closure hierarchies**: close the infinite moment system by choosing an ansatz for the distribution (e.g. Maxwellian, or a non-isotropic Gaussian) so that a finite set of moments (density, momentum, stress, heat flux, …) evolves in a closed way. The “Levermore basis” refers to a choice of moments that respects Galilean invariance and isotropy; the closure gives an **equilibrium** (or quasi-equilibrium) consistent with those moments.

In LBM terms:

- The **equilibrium** is not necessarily the polynomial truncation of the Maxwellian, but the discrete counterpart of the distribution that comes from the chosen moment closure (e.g. Gaussian in velocity).
- This can mean matching **more moments** than the standard polynomial (e.g. higher-order moments or a different stress tensor), or a different functional form altogether.

**Pros:**

- Theoretically grounded in kinetic theory; can improve consistency with Euler/Navier-Stokes (e.g. stress tensor, Galilean invariance) when going beyond the simplest polynomial.
- Can be designed to preserve **hyperbolicity** and **entropy** in the moment system.

**Cons:**

- “Levermore” in LBM is used in several ways (moment-space BGK, central-moment models, etc.); not a single, universal recipe.
- Realizability (moments corresponding to a positive distribution) is nontrivial in multiple dimensions; the closure may need care to keep $f_i \ge 0$.

**Typical use:** When you want better moment consistency (e.g. for compressible or higher-Mach flows) while staying in a moment/BGK-like framework.

**Citation:** Levermore, C.D. (1996). Moment closure hierarchies for kinetic theories. *Journal of Statistical Physics* **83**, 1021-1065. DOI: 10.1007/BF02179552.

---

## 4. Extended equilibrium / extended Levermore

**Idea:** Keep the same lattice and BGK (or BGK-like) structure, but change the **equilibrium** so that the **macroscopic equations** are correct to higher order in Mach number or so that **stress tensor** and **Galilean invariance** are correct for compressible flow. That usually means adding **extra terms** to the standard polynomial equilibrium (e.g. terms that fix the stress tensor or the energy equation), or using an equilibrium derived from an “extended” moment closure (sometimes called extended Levermore in the literature).

So:

- **Extended equilibrium:** $f^{\mathrm{eq}}$ has the usual polynomial part **plus** correction terms so that, when you take moments, you get the right stress, energy, etc. for compressible Navier-Stokes (or Euler).
- **Extended Levermore:** Same idea but framed in the moment-closure / Levermore language: the closure is chosen so that more moments (e.g. stress, heat flux) are correct; the corresponding discrete equilibrium is the “extended” one.

**Pros:**

- Better **Galilean invariance** and **isotropy** of the stress tensor at finite Mach.
- Can go toward **compressible** flow (shock tubes, shock-vortex, etc.) while still using a single- or double-population LBM on standard lattices.

**Cons:**

- More algebra and coding; equilibrium is more complicated than the basic polynomial.
- May still need additional tricks (e.g. numerical equilibria, filtering) for very strong shocks or supersonic turbulence.

**Typical use:** Compressible LBM (Sod tube, shock-vortex, compressible turbulence) where the standard polynomial equilibrium is not enough.

**Citations:** Saadat, M.H., Dorschner, B., Karlin, I.V. (2021). Extended lattice Boltzmann model. *Entropy* **23**, 475. Frapolli, N., Chikatamarla, S.S., Karlin, I.V. (2015). Entropic lattice Boltzmann model for compressible flows. *Physical Review E* **92**, 061301(R).

---

## 5. MRT (Multiple Relaxation Time)

**Idea:** Collision is performed in **moment space** instead of distribution space. Transform $\mathbf{f} \to \mathbf{m}$ (e.g. density, momentum, stress tensor, ghost moments). Relax each moment toward its equilibrium at a **different rate**: conserved moments (density, momentum) stay fixed; shear stress relaxes with a rate that sets viscosity; “ghost” or non-hydrodynamic moments relax with other rates to improve stability and optionally tune bulk viscosity.

So:

- $\mathbf{m} = M\mathbf{f}$, $\mathbf{m}^{\mathrm{eq}} = M\mathbf{f}^{\mathrm{eq}}$ with a fixed matrix $M$.
- Collision: $m_k \to m_k^{\mathrm{eq}} + (1 - s_k)(m_k - m_k^{\mathrm{eq}})$ with $s_k$ the relaxation rate for moment $k$. Conserved $k$ have $s_k = 0$; shear typically $s_\nu$ related to $\tau$; ghosts chosen for stability.
- Then $\mathbf{f}^{\mathrm{out}} = M^{-1}\mathbf{m}$.

**Pros:**

- **Stability** is usually better than single-rate BGK; you can damp ghost modes more strongly without changing viscosity.
- **Tunable bulk viscosity** (separate relaxation for the trace of the stress).
- Same equilibrium as polynomial BGK; only the collision step changes.

**Cons:**

- More parameters (several $s_k$) and more operations per node (transform, relax, inverse transform).
- Implementation is heavier than BGK.

**Typical use:** Production and research codes where stability or bulk-viscosity control matter; often the default in many LBM libraries.

**Citation:** d’Humières, D. (2002). Multiple-relaxation-time lattice Boltzmann models in three dimensions. *Philosophical Transactions of the Royal Society of London A* **360**, 437-451. DOI: 10.1098/rsta.2001.0955.

---

## 6. TRT (Two Relaxation Time)

**Idea:** Simplified MRT with **two** relaxation parameters only. Split the distribution (or its transform) into a **symmetric** part (with respect to the opposite velocity) and an **antisymmetric** part. Relax the symmetric part with one rate and the antisymmetric part with another. With a suitable choice (e.g. “magic” relation between the two rates), you recover the same viscosity as BGK but with better stability and fewer spurious reflections at boundaries.

**Pros:**

- Simpler than full MRT (two rates instead of many).
- Better stability and boundary behavior than BGK in many cases with little extra cost.

**Cons:**

- Less flexibility than full MRT (e.g. no independent bulk viscosity).

**Typical use:** When you want a simple upgrade from BGK without going to full MRT.

**Citation:** Ginzburg, I., Verhaeghe, F., d’Humières, D. (2008). Two-relaxation-time lattice Boltzmann scheme: about parametrization, velocity, pressure and mixed boundary conditions. *Communications in Computational Physics* **3**, 427-478.

---

## 7. Central-moment / cumulant LBM

**Idea:** Work in **central-moment** or **cumulant** space (moments of $(\mathbf{e}_i - \mathbf{u})$ instead of $\mathbf{e}_i$). Equilibrium in that space is often chosen so that only hydrodynamic moments are non-zero; collision relaxes each central moment (or cumulant) toward that equilibrium. Because the basis is tied to the local velocity $\mathbf{u}$, **Galilean invariance** of the resulting equations is improved compared to raw distribution-space or raw moment-space BGK.

**Pros:**

- Better **Galilean invariance**; fewer velocity-dependent errors in the stress tensor.
- Often **more stable** than BGK and sometimes than MRT for comparable cost.
- Clean separation of hydrodynamic and non-hydrodynamic modes.

**Cons:**

- Equilibrium and collision are defined in central-moment/cumulant space; more algebra and code.
- Several variants (central moments vs cumulants; exact vs approximated); not a single standard.

**Typical use:** When you care about Galilean invariance and stability without going to entropic or compressible models.

**Citations:** Geier, M., Pasquali, A., Schönherr, M. (2017). Parametrization of the cumulant lattice Boltzmann method for fourth order accurate diffusion. Part I: derivation and validation. *Journal of Computational Physics* **348**, 862-888. Geier, M., Pasquali, A. (2018). Fourth order Galilean invariance for the lattice Boltzmann method. *Computers & Fluids* **166**, 139-145. (Central-moment LBM: e.g. Premnath, K.N., Banerjee, S. (2009). Incorporating forcing terms in the lattice Boltzmann approach. *Physical Review E* **79**, 026704.)

---

## 8. Regularized LBM

**Idea:** In the collision step, do **not** use the raw $f_i$. Instead, replace $f_i$ by $f_i^{\mathrm{eq}} + f_i^{(1)}$ where $f_i^{(1)}$ is a **reconstructed** non-equilibrium part that lives only in the subspace that affects the stress tensor (e.g. second-order Hermite contribution). So you “regularize” by projecting the non-equilibrium onto the physically relevant part before applying relaxation.

**Pros:**

- **Less numerical noise** and often **better stability**; the irrelevant (ghost) non-equilibrium is discarded.
- Small change to a BGK code: one projection step before collision.

**Cons:**

- Slightly more work per node than plain BGK; need to compute moments and reconstruct $f^{(1)}$.
- Benefits are problem-dependent.

**Typical use:** When you want a simple stabilization and noise reduction without changing to MRT or central-moment.

**Citation:** Latt, J., Chopard, B. (2006). Lattice Boltzmann method with regularized pre-collision distribution functions. *Mathematics and Computers in Simulation* **72**, 165-168. DOI: 10.1016/j.matcom.2006.05.017.

---

## Short comparison

| Scheme              | Equilibrium form        | Main advantage              | Main cost / limitation        |
|---------------------|-------------------------|-----------------------------|--------------------------------|
| **Polynomial**      | Explicit polynomial in u | Simple, fast, good for low Ma | Can go negative; limited to low Ma |
| **Entropic**        | Implicit (minimize H)   | Positivity, H-theorem, stability | Implicit solve; tuning of entropy fix |
| **Levermore**       | Moment-closure based    | Better moment consistency   | Less standardized; realizability |
| **Extended / Ext. Levermore** | Polynomial + corrections | Compressible, Galilean stress | More complex equilibrium and analysis |
| **MRT**             | Same as polynomial      | Stability, tunable bulk viscosity | More parameters and transforms |
| **TRT**             | Same as polynomial      | Simpler MRT, better stability/boundaries | Two rates only |
| **Central-moment / cumulant** | In central-moment space | Galilean invariance, stability | More algebra and variants |
| **Regularized**     | Same as polynomial      | Less noise, simple stabilization | Extra projection step |

---

## References (consolidated)

- **Polynomial / BGK:** Bhatnagar, P.L., Gross, E.P., Krook, M. (1954). A model for collision processes in gases. I. Small amplitude processes in charged and neutral one-component systems. *Physical Review* **94**, 511-525. DOI: 10.1103/PhysRev.94.511. — Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G., Viggen, E.M. (2017). *The Lattice Boltzmann Method*. Springer.
- **Entropic:** Ansumali, S., Karlin, I.V., Öttinger, H.C. (2003). Minimal entropic kinetic models for hydrodynamics. *Europhysics Letters* **63**, 798-804. — Karlin, I.V., Ansumali, S., De Angelis, E., Öttinger, H.C., Succi, S. (2003). Entropic lattice Boltzmann method for large scale turbulence simulation. *International Journal of Modern Physics C* **14**, 1111-1123.
- **Levermore:** Levermore, C.D. (1996). Moment closure hierarchies for kinetic theories. *Journal of Statistical Physics* **83**, 1021-1065. DOI: 10.1007/BF02179552.
- **Extended equilibrium:** Saadat, M.H., Dorschner, B., Karlin, I.V. (2021). Extended lattice Boltzmann model. *Entropy* **23**, 475. — Frapolli, N., Chikatamarla, S.S., Karlin, I.V. (2015). Entropic lattice Boltzmann model for compressible flows. *Physical Review E* **92**, 061301(R).
- **MRT:** d'Humières, D. (2002). Multiple-relaxation-time lattice Boltzmann models in three dimensions. *Philosophical Transactions of the Royal Society of London A* **360**, 437-451. DOI: 10.1098/rsta.2001.0955.
- **TRT:** Ginzburg, I., Verhaeghe, F., d'Humières, D. (2008). Two-relaxation-time lattice Boltzmann scheme: about parametrization, velocity, pressure and mixed boundary conditions. *Communications in Computational Physics* **3**, 427-478.
- **Central-moment / cumulant:** Geier, M., Pasquali, A., Schönherr, M. (2017). Parametrization of the cumulant lattice Boltzmann method for fourth order accurate diffusion. Part I: derivation and validation. *Journal of Computational Physics* **348**, 862-888. — Geier, M., Pasquali, A. (2018). Fourth order Galilean invariance for the lattice Boltzmann method. *Computers & Fluids* **166**, 139-145. — Premnath, K.N., Banerjee, S. (2009). Incorporating forcing terms in the lattice Boltzmann approach. *Physical Review E* **79**, 026704 (central-moment formulation).
- **Regularized:** Latt, J., Chopard, B. (2006). Lattice Boltzmann method with regularized pre-collision distribution functions. *Mathematics and Computers in Simulation* **72**, 165-168. DOI: 10.1016/j.matcom.2006.05.017.
