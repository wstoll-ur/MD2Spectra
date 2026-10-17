"""Frequency-selected MD displacements; these are not normal-mode eigenvectors."""
import argparse
from pathlib import Path
import numpy as np
from ase.io import write
from ase.io.trajectory import Trajectory
from scipy.signal import detrend, windows
from .geometry import molecular_geometry, unwrap_positions, validate_frame
from .io import save_json, sha256
from .spectra import C_CM_S


def filter_motion(positions, times_ps, cell, masses, low_cm1, high_cm1, taper_fraction=.2):
    times_ps = np.asarray(times_ps)
    if len(times_ps) < 16 or not np.isfinite(times_ps).all():
        raise ValueError('At least 16 finite time samples required')
    dt = float(np.median(np.diff(times_ps))) * 1e-12
    if dt <= 0 or not np.allclose(np.diff(times_ps)*1e-12, dt, rtol=1e-7, atol=1e-22):
        raise ValueError('Motion requires uniform, increasing time_ps')
    nyquist = 1 / (2 * dt * C_CM_S)
    if not 0 < low_cm1 < high_cm1 < nyquist:
        raise ValueError(f'Band must satisfy 0 < low < high < Nyquist ({nyquist:.2f} cm^-1)')
    positions = unwrap_positions(np.asarray(positions), cell)
    centre = np.einsum('tij,i->tj', positions, masses) / np.sum(masses)
    positions -= (centre - centre.mean(axis=0))[:, None, :]
    mean = positions.mean(axis=0)
    displacement = detrend(positions, axis=0, type='linear')
    frequencies = np.fft.rfftfreq(len(positions), dt) / C_CM_S
    chosen = (frequencies >= low_cm1) & (frequencies <= high_cm1)
    if not chosen.any():
        raise ValueError('No FFT bins in selected band; use a longer trajectory or wider band')
    # Raised-cosine edges inside the requested band; no frequencies outside it.
    weight = chosen.astype(float)
    edge = (high_cm1 - low_cm1) * .15
    lo = chosen & (frequencies < low_cm1 + edge)
    hi = chosen & (frequencies > high_cm1 - edge)
    weight[lo] = .5 * (1 - np.cos(np.pi * (frequencies[lo] - low_cm1) / edge))
    weight[hi] = .5 * (1 - np.cos(np.pi * (high_cm1 - frequencies[hi]) / edge))
    if not np.any(weight > 0):
        raise ValueError('Band has no nonzero filter bins; widen it or use a longer trajectory')
    transform = np.fft.rfft(displacement * windows.tukey(len(positions), taper_fraction)[:, None, None], axis=0)
    filtered = np.fft.irfft(transform * weight[:, None, None], n=len(positions), axis=0)
    return mean, filtered, float(frequencies[1]), nyquist


def export_motion(trajectory, output, low, high, unit='cm-1', start_ps=None, end_ps=None,
                  amplitude=5., clip_ps=.5, stride=4, bond_scale=1.15):
    trajectory, output = Path(trajectory), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Motion output already exists; choose a new filename')
    if not all(np.isfinite(v) and v > 0 for v in [amplitude, clip_ps, bond_scale]) or stride < 1:
        raise ValueError('Amplitude, clip length, stride and bond scale must be positive')
    if start_ps is not None and (not np.isfinite(start_ps) or start_ps < 0):
        raise ValueError('start-ps must be nonnegative')
    if end_ps is not None and (not np.isfinite(end_ps) or end_ps <= (start_ps or 0)):
        raise ValueError('end-ps must exceed start-ps')
    if unit == 'THz':
        low, high = low * 1e12 / C_CM_S, high * 1e12 / C_CM_S
    source_hash = sha256(trajectory)
    with Trajectory(trajectory) as frames:
        indices = []
        for i, atoms in enumerate(frames):
            t = atoms.info['time_ps']
            if (start_ps is None or t >= start_ps) and (end_ps is None or t <= end_ps):
                indices.append(i)
        if len(indices) < 16:
            raise ValueError('Time selection has fewer than 16 frames')
        reference = frames[indices[0]]
        groups, whole, bonds = molecular_geometry(reference, bond_scale)
        positions = np.empty((len(indices), len(reference), 3))
        times = np.empty(len(indices))
        for k, i in enumerate(indices):
            atoms = frames[i]
            validate_frame(atoms, reference)
            g, _, current_bonds = molecular_geometry(atoms, bond_scale)
            if current_bonds != bonds or not np.array_equal(g, groups):
                raise ValueError(f'Molecular connectivity changed at frame {i}')
            positions[k], times[k] = atoms.positions, atoms.info['time_ps']
    # One fixed lattice translation per atom makes the first frame whole;
    # temporal unwrapping then retains continuity throughout the selection.
    positions += whole - reference.positions
    mean, motion, spacing, nyquist = filter_motion(positions, times, reference.cell,
        reference.get_masses(), low, high)
    dt_ps = float(np.median(np.diff(times)))
    if stride * dt_ps * 1e-12 * high * C_CM_S >= .5:
        raise ValueError('Animation stride aliases the selected band; reduce --stride')
    count = round(clip_ps / dt_ps)
    if count < 2 * stride or count > .8 * len(times):
        raise ValueError('Clip must provide at least two output frames and fit inside central 80% of the time selection')
    begin = (len(times) - count) // 2
    selected = np.arange(begin, begin + count, stride)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp.extxyz')
    with temp.open('w') as handle:
        for k in selected:
            atoms = reference.copy()
            atoms.calc = None
            for key in list(atoms.arrays):
                if key not in ['numbers', 'positions']:
                    atoms.arrays.pop(key)
            atoms.positions = mean + amplitude * motion[k]
            atoms.set_array('displacement_A', motion[k])
            atoms.set_array('atom_id', np.arange(1, len(atoms) + 1))
            atoms.set_array('molecule_id', groups)
            atoms.info = dict(time_ps=float(times[k]), source_frame=int(indices[k]),
                              band_low_cm1=low, band_high_cm1=high,
                              amplitude_scale=amplitude, visualization_only=True)
            write(handle, atoms, format='extxyz')
    if sha256(trajectory) != source_hash:
        temp.unlink(missing_ok=True)
        raise ValueError('Source trajectory changed during motion extraction')
    temp.replace(output)
    metadata = dict(source_sha256=source_hash, output_sha256=sha256(output),
                    band_cm1=[low, high], source_frames=[indices[0], indices[-1]],
                    source_time_ps=[float(times[0]), float(times[-1])], fft_spacing_cm1=spacing,
                    nyquist_cm1=nyquist, exported_frames=len(selected), amplitude_scale=amplitude,
                    clip_time_ps=[float(times[selected[0]]), float(times[selected[-1]])],
                    filter='15% raised-cosine band edges; Tukey 0.2 time window; linear detrending; COM drift removal',
                    interpretation='All MD motion within the selected frequency band, including optically inactive motion. Not a unique Raman/IR mode or normal-mode eigenvector.')
    save_json(output.with_suffix('.json'), metadata)
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description='Export selected Raman/THz/IR frequency-band motion as extended XYZ')
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--band', type=float, nargs=2, required=True, metavar=('LOW', 'HIGH'))
    p.add_argument('--unit', choices=['cm-1', 'THz'], default='cm-1')
    p.add_argument('--start-ps', type=float)
    p.add_argument('--end-ps', type=float)
    p.add_argument('--amplitude', type=float, default=5.)
    p.add_argument('--clip-ps', type=float, default=.5)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--bond-scale', type=float, default=1.15)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        export_motion(args.trajectory, args.output, *args.band, args.unit, args.start_ps,
                      args.end_ps, args.amplitude, args.clip_ps, args.stride, args.bond_scale)
    except (ValueError, FileNotFoundError, KeyError) as error:
        p.exit(2, f'Error: {error}\n')
    print(f'Wrote {args.output.resolve()}')


if __name__ == '__main__':
    main()
