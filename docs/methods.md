# Methods and scope

## Dynamics

ASE integrates UMA forces using the Bussi stochastic velocity-rescaling NVT
thermostat. Internal-coordinate FIRE relaxation is optional; the CIF cell remains
fixed. Equilibration and production lengths are independent. New runs sample
Maxwell–Boltzmann velocities and remove centre-of-mass momentum. TF32 and Torch
compilation are disabled for UMA. Atomic isotope masses are not currently a CLI
option. NPT, charged systems, reactive trajectories and automated cluster
submission are not implemented.

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
