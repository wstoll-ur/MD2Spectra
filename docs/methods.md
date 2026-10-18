# Methods and scope

## Dynamics

ASE integrates UMA forces using the Bussi stochastic velocity-rescaling NVT
thermostat. Internal-coordinate FIRE relaxation is optional; each trajectory's
cell remains fixed at the supplied geometry or its QHA-prepared temperature cell.
Equilibration and production lengths are independent. New runs sample
Maxwell–Boltzmann velocities and remove centre-of-mass momentum. TF32 and Torch
compilation are disabled for UMA. Atomic isotope masses are not currently a CLI
option. NPT, charged systems, reactive trajectories and automated cluster
submission are not implemented.

## Quasi-harmonic cell preparation

The optional QHA stage follows the [volume-dependent harmonic free-energy
definition](https://phonopy.github.io/phonopy/qha.html), implemented with
[ASE finite-displacement phonons](https://docs.ase-lib.org/ase/phonons.html).
The input crystallographic cell is used as the phonon basis; no primitive-cell
reduction is required. At each volume factor `s = V/V_input`, lattice vectors
are scaled by `s^(1/3)` and internal coordinates are relaxed with FIRE under UMA.
Cell shape and angles stay fixed during this scan. QHA preparation, including
the subsequent temperature-cell shape optimization, is always performed,
independent of the optional MD `--no-relax` setting.

Central displacements generate real-space force constants in the independent
`--qha-supercell`. ASE symmetrization and the acoustic sum rule are applied.
Phonon energies `epsilon(q,j) = hbar*omega(q,j)` are in eV. A shifted
Monkhorst–Pack grid with even dimensions avoids Gamma's translational zero
modes. Every mode on that grid must have positive energy. Gamma is checked
separately: the lowest three branches must be within 1e-4 eV of zero, and all
remaining modes must be positive. Negative ASE energies represent imaginary
frequencies; they are never replaced by absolute values or omitted.

For each requested temperature:

```text
F_vib(V,T) = (1/Nq) sum_(q,j) [epsilon/2 + kB*T*ln(1-exp(-epsilon/(kB*T)))]
G(V,T,p) = E_static(V) + F_vib(V,T) + p*V
V_eq(T,p) = argmin_V G(V,T,p)
```

The T=0 expression contains only zero-point vibrational energy. Equal q-point
weights normalize the sum to the **input cell**, the same normalization used for
static energies and volumes. Positive pressure favors smaller volumes; GPa is
converted to eV/Å³ with ASE units. Free-energy arrays have shape
`(number_of_volumes, number_of_temperatures)`.

A cubic polynomial is fitted to each temperature's free-energy curve on at least
five volume points. The sampled minimum must be interior, the fit must be convex
throughout the sampled interval, and its stationary point must be bracketed by
the sampled minimum's neighboring volumes. No extrapolation is allowed. A
bracketed derivative root gives the equilibrium volume. Fit residuals and
`B = V*d²G/dV²` are exported for assessment; a successful fit is not a
convergence criterion. Check residuals against the energy differences that
determine the minimum and vary the grid to establish fit stability.

For each fitted volume, the nearest sampled relaxed structure is isotropically
rescaled, then its **cell shape and atomic positions are optimized together at
that fixed volume**. FIRE uses an ASE Frechet cell filter with volume-projected
forces. Each cell-coordinate update also preserves the reference trace of the
logarithmic deformation, enforcing `det(F) = exp(trace(log(F))) = 1` to numerical
precision. This allows all volume-preserving changes in lengths and angles,
including shear. Every step must retain the fitted volume to relative tolerance
1e-10. Optimization requires both maximum atomic force below `--qha-fmax` and
maximum absolute component of deviatoric stress below `--qha-shape-stress-gpa`
(default 0.01 GPa). Hydrostatic stress is not driven to zero at constrained volume.
The calculator must provide stress. Failed convergence stops preparation before MD.

The saved extended XYZ contains the optimized full cell matrix and positions;
its JSON sidecar records target/actual volume, energy, lengths, angles and
convergence metrics. Existing MD preparation repeats this optimized cell to build
the requested MD supercell. Its full cell matrix remains fixed during NVT.
Each temperature starts independent velocities, equilibration and production;
no trajectory is relabeled at a different temperature. Spectral analysis receives
the actual trajectory volume and corresponding thermostat temperature.

The volume fit samples hydrostatic scaling at the input cell shape; the later
shape optimization minimizes static UMA energy at each fitted volume. It does
not include vibrational free-energy derivatives with respect to strain or
recompute the phonon spectrum and volume fit at the optimized shapes. This is
therefore a sequential QHA-volume/static-shape preparation, not a fully
anisotropic QHA treatment, and can miss vibrational strain effects and feedback
of shape relaxation on equilibrium volume in molecular crystals.
It assumes the same stable phase throughout the scan, and is not a treatment
of melting, reconstructive transitions or strongly anharmonic soft modes.
Long-range nonanalytic dipole corrections are not added to the force constants.
The phonon supercell, q mesh, displacement, force accuracy and volume range must
be converged for the selected potential and material. The MD trajectory remains
classical even though the QHA volume includes quantum harmonic free energy.

## Raman

Let `a(t) = trace(alpha(t))/3` and `B(t) = alpha(t) - a(t) I`. With one-sided
Welch power spectral densities per Hz:

```text
S_a = PSD(a)
S_gamma = (3/2) sum_ij PSD(B_ij)
parallel = (45 S_a + 4 S_gamma) / V
crossed  = 3 S_gamma / V
classical total = (45 S_a + 7 S_gamma) / V
```

The total system polarizability is transformed before dividing the resulting
power by volume. Dividing alpha by volume before computing its power would give
a different, inappropriate size normalization here. Channels have units
`(e Angstrom²/V)² / (Hz Angstrom³)`. There is no absolute scattering prefactor.

The displayed Stokes response is multiplied by
`Q = x / (1 - exp(-x))`, where `x = hc*wavenumber/(kB*T)`, and by
`max(1 - wavenumber/laser_wavenumber, 0)^4`. This is an optional harmonic quantum
correction and relative optical factor. Both corrected and classical data are
retained. It does not recover zero-point motion or quantum frequency shifts.
The calculation is a powder average, not a specific single-crystal scattering
geometry. CSV densities remain per Hz even when plotted against cm⁻¹; multiplying
by the speed of light in cm/s converts to densities per cm⁻¹ for integration.

## Periodic dipoles and IR

The model's raw dipole contains coordinate-dependent `sum_i q_i r_i` terms.
Wrapping atoms into a periodic cell changes this branch. Applying a Fourier
transform directly to that raw series can create spurious IR intensity.

For finite, neutral molecules M, this implementation evaluates:

```text
mu_molecular = mu_raw - sum_i q_i r_i
               + sum_M sum_(i in M) q_i (r_i_whole - centre_M)
```

`centre_M` is the unweighted mean of the whole molecular coordinates. A uniform
charge offset cancels in each centred molecule, so globally normalized MACE
charges suffice. The atomic dipole contributions already contained in `mu_raw`
are retained. Translating a whole molecule by a lattice vector does not change
this definition. Molecular topology is inferred with ASE covalent radii, checked
at every labeled frame and recorded for inspection. Wrapping and rotation checks
run on the actual response calculator before labeling.

**This is a neutral-molecule approximation.** It excludes intermolecular charge
transfer and ionic transport. It can represent a neutral zwitterionic molecule
such as glycine, but not a salt by independently neutralizing its ions. The graph
cannot determine chemical net charges: users must verify that each inferred
component is a complete neutral molecule. It rejects periodic bonded networks
and changing connectivity. `--bond-scale` exposes the covalent cutoff assumption.
Do not use this implementation as a general polarization solution for metals,
ionic conductors, covalent networks or reactive chemistry.

The relative absorption channels are:

```text
IR_i(omega) = omega² PSD(mu_i) / (T V)
IR_powder = (IR_x + IR_y + IR_z) / 3
```

Physical constants and the refractive index are omitted, so this is an absorption
proxy, not a calibrated coefficient in cm⁻¹. In the classical/harmonic-QCF
approximation, detailed balance cancels the extra quantum occupation factor in
absorption. The Raman Stokes QCF is therefore **not multiplied into IR again**.
The PSD of raw dipole, without omega² weighting, is not the absorption spectrum.
THz and IR here are the same electric-dipole response on different frequency
ranges. Magnetic-dipole absorption and higher-order light interactions are absent.

## Sampling and interpretation

Saving every 1 fs resolves frequencies up to about 16,678 cm⁻¹. The pipeline
checks the requested spectral range against Nyquist. It cannot prove that
unrepresented higher-frequency motion is absent. Duration, window broadening,
thermostat coupling, potential accuracy and molecular response-model transferability
all affect interpretation. Use independent trajectories and multiple cell sizes
for convergence. A short smoke test verifies execution, not spectral accuracy.

An animation retains all displacement Fourier components in its selected band,
with raised-cosine edges and a Tukey time taper. It is not weighted by optical
activity. A unique eigenvector requires a separate normal-mode analysis; it is
not inferred by this exporter. Long-time diffusion and reactions violate the
vibration-around-a-reference interpretation. The saved-frame interval must be
short enough that minimum-image temporal unwrapping is unambiguous.
