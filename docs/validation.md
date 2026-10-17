# Validation record

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
