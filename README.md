# MD2Spectra

Turn a crystal structure into **UMA molecular dynamics, powder Raman spectra,
THz/IR absorption proxies, and frequency-selected atomic-motion animations**.

Two commands:

- `md2spectra run`: CIF → supercell → fixed-cell relaxation → equilibration → production MD → response labeling → spectra.
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
python -m pip install '.[uma,response]'
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
separately requested equilibration time. The cell stays fixed at the supplied
experimental geometry; `--supercell` repeats it without changing its density.

Add `--dry-run` to inspect atom count, inferred molecular sizes, volume, resolution,
and frame count without loading either model or launching MD. Use `--cif-index`
for a zero-based structure index in a CIF containing several data blocks.
Partial occupancies must be resolved into an ordered structure first.

Useful options include `--seed`, `--thermostat-fs`, `--checkpoint-ps`,
`--no-relax`, `--fmax`, `--response-device cpu`, and `--bond-scale`.
Run `md2spectra run --help` for the full interface. A synthetic water-box CIF is
provided in `examples/` solely for testing the interface; it is not an
experimental crystal or a reference spectrum.

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
