# Validation record

QHA extension validation, 2026-10-18:

- All 26 unit/integration tests passed, including the 11 existing regressions.
- Analytic checks cover zero-point energy, quantum harmonic free energy,
  q-point normalization, thermal expansion from mode softening, pressure units,
  and equilibrium-volume fitting with a large static energy offset.
- Real ASE finite-displacement phonons with the EMT copper potential exercise
  the volume scan and temperature-cell export. Short EMT MD confirms that the
  selected QHA volume, multiplied by the requested MD supercell size, stays fixed
  in every trajectory frame.
- Restart checks reuse completed volumes after interruption, reject changed
  settings and detect modified output cells. CLI checks cover inclusive ranges,
  explicit temperature lists, invalid inputs, standalone 0 K QHA, per-temperature
  cell/temperature dispatch, and stopping before MD after a failed QHA stage.
- The documented seven-temperature dry run and standalone QHA help command passed.
- Constant-volume shape tests use distorted copper cells to check relaxation of
  lengths, angles and internal positions, energy reduction, force/stress
  convergence and volume preservation on every evaluated configuration. Numerical
  energy derivatives check the projected cell-filter forces. Tests also cover
  shape restart/settings protection, rejection of unconverged cells and
  preservation of the optimized full cell matrix throughout NVT.

These tests validate implementation and data flow. A production UMA QHA scan
and material-specific convergence/experimental validation have **not** been run.
The phonon volume scan preserves cell shape; a static constant-volume shape
optimization follows at each temperature. Full anisotropic QHA is outside its
scope. See [the QHA methods](methods.md#quasi-harmonic-cell-preparation).

Initial local validation, 2026-10-17:

- Ten model-independent unit/integration tests passed. They include an
  interrupted Bussi trajectory, an extra uncommitted frame, restart, and comparison
  with an uninterrupted harmonic reference at absolute tolerance 1e-14 for both
  positions and momenta. This tests checkpoint logic, not UMA physical accuracy.
- Synthetic response labeling, reuse of label chunks, spectrum generation and
  extended-XYZ readback passed. Numerical tests check the Raman tensor invariants,
  IR frequency-squared weighting, known frequencies and requested-band recovery.
- The pinned real MACE-MDP checkpoint passed periodic wrapping and rotation
  qualification on the existing 40-atom glycine seed. Maximum errors were
  3.52e-15 (wrapped alpha), 6.91e-15 (wrapped molecular dipole),
  6.44e-15 (rotated alpha), and 1.38e-14 (rotated molecular dipole), in model units.
  The model SHA-256 was verified before loading.
- A real-model labeling and spectrum-export smoke test passed on 16 existing
  glycine trajectory frames, sampled every 10 fs. This validates execution of the
  response pipeline; its short duration and 1,668 cm⁻¹ Nyquist limit make it
  unsuitable for interpreting full vibrational spectra.
- A wheel built and installed successfully in a separate environment. Both
  command entry points and a four-molecule supercell dry run passed; all ten
  regression tests also passed against the installed package.

The local reference environment was Python 3.11, ASE 3.29.0, NumPy 2.4.6,
SciPy 1.17.1, MACE 0.3.15 and Torch 2.13.0. CUDA inference can produce different
roundoff. The packaging pins UMA's adapter dependency to FAIRChem 2.23.0 and the
response adapter to MACE 0.3.15; the installed FAIRChem loader interfaces were
checked against that version.

A fresh end-to-end **UMA** production calculation has not been run as part of
this repository preparation. It requires the gated UMA checkpoint and an
appropriate compute environment. No benchmark spectrum, physical accuracy or
converged production result is claimed by these tests. The neutral-molecule IR
approximation also needs validation against appropriate reference spectra.

To qualify a trusted response model locally:

```bash
python tests/qualify_mdp.py \
  --model /path/to/MACE-MDP.model \
  --structure crystal.cif --device cpu
```

This also checks extensivity when repeating the cell. To exercise UMA cheaply,
first use the dry run, then a short, explicitly nonproduction calculation:

```bash
md2spectra run --cif crystal.cif --temperature 300 \
  --equilibration-ps 0.01 --production-ps 0.02 --checkpoint-ps 0.01 \
  --segment-ps 0.01 --device cuda --output runs/smoke
```

Such a short trajectory is only an execution check. Use much longer trajectories,
equilibration assessment, independent sampling and finite-size tests for science.
