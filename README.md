# MD2Spectra

Turn a crystal structure into **UMA molecular dynamics, powder Raman spectra,
THz/IR absorption proxies, and frequency-selected atomic-motion animations**.

Main commands:

- `md2spectra run`: CIF → supercell → fixed-cell relaxation → equilibration → production MD → response labeling → spectra.
- `md2spectra run --qha`: CIF → QHA volumes → constant-volume cell-shape optimization → one NVT/response/spectrum run per temperature.
- `md2spectra qha`: calculate temperature-dependent volumes and cells without launching MD.
- `md2motion`: select a frequency interval and export its MD motion as an extended XYZ movie.

The initial implementation targets **ordered, neutral molecular crystals**, such
as glycine. UMA supplies energies and forces; **MACE-MDP supplies dipoles and
polarizabilities**. THz is the low-frequency portion of the IR spectrum, also
reported on a THz axis. Intensities are relative; this is not an absolute
experimental cross-section calculation.

## Installation

Python 3.11 or newer; Linux or macOS. A CUDA GPU is strongly recommended for
production dynamics. CPU execution is available with `--device cpu`.

```bash
git clone https://github.com/wstoll-ur/MD2Spectra.git
cd MD2Spectra
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[uma,response,qha]'
```

Request access to [facebook/UMA](https://huggingface.co/facebook/UMA), accept its
model terms, and authenticate once using `hf auth login`. The default UMA model
is `uma-s-1p2p1`, with the `omc` task for organic molecular crystals. A local
checkpoint can instead be supplied with `--uma-model /path/to/checkpoint.pt`.

The pinned [MACE-MDP model](https://github.com/Nilsgoe/MACE-MDP) downloads on first
use to `~/.cache/md2spectra`; its SHA-256 is checked before loading. Its authors
provide it under the Academic Software License for academic, noncommercial use.
Use `--mdp-model /path/to/MACE-MDP.model` for offline operation or a compatible,
trusted custom checkpoint. Model weights are not redistributed in this repository.

UMA and MACE run in separate subprocesses, releasing model memory between stages.
If dependency requirements conflict on your machine, install `.[uma]` in the main
environment and `.[response]` in a second environment, then pass
`--response-python /path/to/response-venv/bin/python`. Install this package in
**both** environments. Spectra and motion analysis need only `pip install .`.

## One command from a CIF

```bash
md2spectra run \
  --cif crystal.cif \
  --temperature 300 \
  --supercell 2 1 2 \
  --equilibration-ps 10 \
  --production-ps 100 \
  --timestep-fs 0.25 \
  --save-every 4 \
  --segment-ps 20 \
  --laser-nm 532.1 \
  --device cuda \
  --output runs/crystal-300K
```

This saves frames every **1 fs**, giving a Nyquist frequency of approximately
16,678 cm⁻¹. Twenty-picosecond Hann/Welch segments give approximately 1.668 cm⁻¹
bin spacing; the window also broadens peaks. `--production-ps` excludes the
separately requested equilibration time. By default, the cell stays fixed at the
supplied experimental geometry; `--supercell` repeats it without changing its density.

Add `--dry-run` to inspect atom count, inferred molecular sizes, volume, resolution,
and frame count without loading either model or launching MD. Use `--cif-index`
for a zero-based structure index in a CIF containing several data blocks.
Partial occupancies must be resolved into an ordered structure first.

Useful options include `--seed`, `--thermostat-fs`, `--checkpoint-ps`,
`--no-relax`, `--fmax`, `--response-device cpu`, and `--bond-scale`.
Run `md2spectra run --help` for the full interface. A synthetic water-box CIF is
provided in `examples/` solely for testing the interface; it is not an
experimental crystal or a reference spectrum.

### Temperature series with QHA volumes

Select QHA and the MD temperature range in the initial command:

```bash
md2spectra run \
  --cif crystal.cif \
  --qha --temperature-range 100 400 50 \
  --qha-volume-factors 0.94 0.96 0.98 1.00 1.02 1.04 1.06 \
  --qha-supercell 2 2 2 --qha-mesh 8 8 8 \
  --supercell 2 1 2 \
  --equilibration-ps 10 --production-ps 100 --segment-ps 20 \
  --raman-only --device cuda \
  --output runs/crystal-qha
```

`--temperature-range START STOP STEP` includes both endpoints; STOP must be
exactly reachable by stepping from START. This example runs at 100, 150, 200,
250, 300, 350 and 400 K. Use `--temperatures 100 175 300` for an explicit list,
or `--temperature 300 --qha` for a single temperature. These choices are mutually
exclusive. MD temperatures must be positive and lists strictly increasing.

QHA uses UMA static energies and finite-displacement phonons at each sampled
volume to minimize `E_static(V) + F_vib(V,T) + pV`, including zero-point energy.
Install the `qha` extra in the UMA environment. ASE collects restartable force
calculations; Phonopy constructs and symmetrizes the force constants and uses
pair-dependent shortest vectors for interpolation across the q mesh.
The phonon scan is shared by all temperatures. At each fitted volume, a
**constant-volume optimization relaxes lattice lengths, angles and atomic
positions** using UMA energy, forces and stress. Each MD run then equilibrates
independently at that optimized cell and its own temperature, followed by response
labeling and Raman analysis. The trajectory volume and temperature are passed
through to spectral normalization and quantum correction automatically.

The **QHA volume scan** preserves the input cell's shape and relaxes internal
coordinates. The subsequent per-temperature shape optimization holds the fitted
volume fixed while allowing all deviatoric strains, including shear. The resulting
full cell matrix stays fixed throughout NVT. Shape relaxation minimizes static
energy; phonon free energies are not recomputed for the changed shapes, so this
is not a full anisotropic QHA minimization. QHA volumes include quantum harmonic
effects, while subsequent MD remains classical. See
[methods and limitations](docs/methods.md#quasi-harmonic-cell-preparation).

Volume factors are `V/V_input`, not lattice-length factors. The default range is
a starting grid, not a guaranteed equilibrium bracket. An unstable phonon spectrum,
nonconvex fit, or unbracketed minimum stops the workflow before MD. Inspect the
saved free energies and converge the volume grid, phonon supercell, q-point mesh,
displacement and relaxation threshold for the crystal of interest. No unstable
modes or volume points are silently removed. Refine settings in a new directory.

`--qha-supercell` controls force-constant calculations independently of the MD
`--supercell`; `--qha-mesh` specifies even dimensions of a shifted q-point mesh.
Other options are `--qha-displacement` (default 0.01 Å), `--qha-fmax` (default
0.001 eV/Å), `--qha-shape-stress-gpa` (default 0.01 GPa, maximum absolute
deviatoric-stress component), and `--qha-pressure-gpa` (default 0). Both force and
deviatoric-stress tolerances must be met before a temperature cell is accepted;
hydrostatic stress is allowed because volume is constrained. Volume is checked
on every optimization step. QHA always uses conservative
UMA inference because it evaluates both the input cell and phonon supercells;
`--uma-inference md` still selects the optimized profile for subsequent MD.
No additional dependency beyond the existing ASE/UMA installation is needed.

Add `--dry-run` to validate the inputs and preview the scan without model loading.
Dry runs report the input volume; predicting temperature-dependent volumes
requires executing QHA. To run QHA alone, including an optional 0 K point:

```bash
md2spectra qha --cif crystal.cif --temperature-range 0 400 50 \
  --device cuda --output runs/crystal-qha-only
```

The temperature-series output contains:

```text
runs/crystal-qha/
  series.json                        initial settings and source checksum
  qha/
    qha.json                         scan identity and model checksum
    volumes.csv                      temperature, input-cell volume, fit diagnostics
    free-energy.npz                  static/vibrational/total energies on the grid
    volume-000/                      relaxed cell, restartable force cache, phonons
    cells/temperature-000/relaxed.extxyz
    cells/temperature-000/relaxed.json  fitted volume, cell geometry, convergence checks
    qha-complete.json                 temperature-to-cell mapping and checksums
  temperature-000-100K/               full MD/response/spectrum outputs
  temperature-001-150K/
  ...
  complete.json                      written after every temperature succeeds
```

Repeat the identical command to resume. Completed phonon displacements, volume
calculations and temperature runs are reused; model/input/settings changes are
rejected. Outputs from earlier QHA implementations, including ASE-only phonon
processing, require a new output directory. Each temperature directory retains the existing MD checkpoint, `STOP`
and labeling restart behavior. QHA's interrupted displacement resumes from its
force cache; interrupted relaxation repeats that volume's relaxation. QHA,
MD and response stages run sequentially in separate processes to release model
memory between stages.

### Outputs and restart

```text
runs/crystal-300K/
  md.json                    settings, runtime versions, input/model SHA-256
  initial.extxyz             prepared/relaxed structure
  trajectory.traj            positions, momenta, steps and time_ps
  checkpoint.json            coordinates, momenta, RNG and thermostat state
  progress.json              latest committed MD progress
  md-complete.json           completed trajectory checksum
  response/
    labels-manifest.json     model/source hashes, molecular IDs and bonds
    qualification.json       periodic wrapping and rotation checks
    chunk-*.npz              resumable response predictions
    labels.npz               times, polarizabilities, molecular/raw dipoles
    labels-complete.json     completed response checksum
  spectra/
    spectra.csv              frequency axes and all intensity channels
    spectra.png              Raman and IR overview
    spectrum.json            conventions, sampling and resolution
    block-*.csv              three sampling blocks, when the run is long enough
  complete.json              successful completion of the full pipeline
```

Rerun the **same command** to continue a stopped run. MD restores positions,
momenta, Bussi thermostat state and random-number state, and removes uncommitted
trajectory tails. Source/model/settings changes are rejected; choose a new output
directory for a new simulation. MD checkpoints are written every 1 ps by default.
Completed label chunks are reused. Errors stop the pipeline instead of proceeding
to a partial spectrum. Each stage holds a process lock.

SIGINT/SIGTERM requests a checkpoint at the next saved frame. Creating a `STOP`
file in the run directory stops MD similarly; remove it before resuming.
Exit code 75 means an intentional early MD stop. Abrupt termination can require
replaying steps after the last checkpoint. Filesystem/storage corruption is not
repaired automatically. GPU restarts may differ slightly numerically even with
restored stochastic state.

`complete.json` records a successful calculation, not scientific convergence.
Check equilibration, duration, time blocks, supercell size and model accuracy.

## Export a frequency band as extended XYZ

Use the calculated spectrum to choose a frequency interval, then select motion
from the matching trajectory:

```bash
md2motion \
  --trajectory runs/crystal-300K/trajectory.traj \
  --band 475 500 --unit cm-1 \
  --start-ps 10 --end-ps 30 \
  --clip-ps 0.5 --stride 4 --amplitude 8 \
  --output runs/crystal-300K/band-475-500.extxyz
```

A THz selection uses the same exporter:

```bash
md2motion \
  --trajectory runs/crystal-300K/trajectory.traj \
  --band 1.2 2.2 --unit THz \
  --start-ps 10 --end-ps 30 \
  --clip-ps 1 --stride 4 --amplitude 8 \
  --output runs/crystal-300K/band-1.2-2.2THz.extxyz
```

The exporter reconstructs whole molecules, unwraps positions in time, removes
centre-of-mass drift, detrends, applies a time taper and a smooth frequency filter,
then exports a central excerpt. Atomic IDs, cells, source times, molecule IDs,
physical filtered displacements and display amplification are preserved. A JSON
sidecar records provenance and the exact passband. Existing exports are protected
from accidental overwrite. Reduce `--stride` for high-frequency bands.

**A frequency band can contain multiple modes and optically inactive motion.**
These are band-filtered MD animations, not unique normal-mode eigenvectors or
proof that every visible displacement causes the Raman/IR peak. The amplitude
multiplier is for visualization; the saved `displacement_A` field is unscaled.
Analysis loads the selected trajectory interval into memory; use `--start-ps`
and `--end-ps` to limit memory use. No spectrum alone can reconstruct atom motion:
the matching trajectory is required.

## Analyze existing trajectories

A trajectory must be an ASE `.traj` with fixed cells, consistent atomic ordering,
and uniformly increasing `time_ps` metadata on every frame.

```bash
md2spectra label --trajectory trajectory.traj --device cuda --output response
md2spectra spectrum --labels response/labels.npz --temperature 300 \
  --discard-ps 10 --segment-ps 20 --output spectra
```

The `md` subcommand runs only the dynamics stage with the same simulation options
as `run`. See [the methods](docs/methods.md) for equations and interpretation,
and [the cluster example](examples/slurm.sh) for a scheduler template.

## Validation

```bash
python -m pip install '.[qha]'
python -m unittest discover -s tests -v
```

Tests cover known frequencies, Raman rotational invariance and polarization,
IR frequency-squared weighting, temperature/volume scaling, periodic molecular
dipoles, invalid network geometries, stochastic restart after an interrupted
trajectory, resumable labeling, spectral export and band-motion recovery.
CI runs the model-independent tests on Python 3.11 and 3.12. Model tests are
optional because UMA requires gated access; see [validation](docs/validation.md)
for the distinction between synthetic regression checks and model qualification.

## Provenance

This package generalizes the existing UMA/MACE-MDP glycine Raman workflow and
frequency-filtered MD animation workflow. Machine-specific scheduling, credentials,
private communications, downloaded papers and bulk campaign trajectories are not
part of the portable package. Original campaign data remain in their source
workspace. See [third-party references](THIRD_PARTY.md).

## Raman-only and long cluster runs

Use `--raman-only` on `run`, `label`, or `spectrum` to omit dipole/IR processing.
Raman-only labeling does not require a molecular dipole convention or fixed
molecular connectivity. Polarizability wrapping and rotation checks still run.
For long jobs, `md --max-wall-s 81000` and
`label --raman-only --max-wall-s 81000` stop with exit code 75 after saving
restart state. Use a sequential scheduler continuation to resume the same stage.
`--no-save-momenta` reduces trajectory storage; momenta remain in every MD
checkpoint, so restart dynamics are unchanged. Gracefully checkpointed trajectories
are reused in place rather than rewritten on every allocation.

For fixed-composition UMA MD, `--uma-inference md` merges composition-dependent
experts and disables activation checkpointing. It retains FP32 with TF32 and
compilation disabled; FAIRChem selects the compatible execution backend.
Benchmark speed and GPU memory on the full supercell before a long campaign.
The conservative profile remains the default. An explicitly selected profile
is part of the restart identity and cannot be changed within an existing run.
Trajectories are retained after spectra are calculated and can be reused for
coordinate-based collective variables and free-energy analysis.
