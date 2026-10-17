"""Welch powder Raman response and classical/HQCF infrared absorption proxy."""
from pathlib import Path
import numpy as np
from scipy.signal import welch
from .io import save_json, sha256

C_CM_S = 2.99792458e10


def calculate(alpha, dipole, dt_fs, temperature, volume, segment_ps=20., laser_nm=532.1):
    alpha = np.asarray(alpha)
    if dipole is not None:
        dipole = np.asarray(dipole)
    if alpha.ndim != 3 or alpha.shape[1:] != (3, 3) or (dipole is not None and dipole.shape != (len(alpha), 3)):
        raise ValueError('Expected N x 3 x 3 alpha and N x 3 dipole')
    if not np.isfinite(alpha).all() or (dipole is not None and not np.isfinite(dipole).all()):
        raise ValueError('Nonfinite response data')
    if not np.allclose(alpha, alpha.transpose(0, 2, 1), rtol=1e-7, atol=1e-10):
        raise ValueError('Polarizability must be symmetric')
    if not all(np.isfinite(x) and x > 0 for x in [dt_fs, temperature, volume, segment_ps, laser_nm]):
        raise ValueError('Sampling interval, temperature, volume, segment and laser must be positive')
    nseg = round(segment_ps * 1000 / dt_fs)
    if nseg < 8 or nseg > len(alpha):
        raise ValueError(f'Correlation window requires {nseg} samples; have {len(alpha)} (minimum window: 8)')
    kw = dict(fs=1e15 / dt_fs, window='hann', nperseg=nseg, noverlap=nseg // 2,
              detrend='constant', scaling='density', axis=0)
    isotropic = np.trace(alpha, axis1=1, axis2=2) / 3
    deviator = alpha - isotropic[:, None, None] * np.eye(3)
    freq, si = welch(isotropic, **kw)
    _, sd = welch(deviator.reshape(len(alpha), 9), **kw)
    sg = 1.5 * sd.sum(axis=1)
    wn = freq / C_CM_S
    x = 1.438776877 * wn / temperature
    qcf = np.ones_like(x)
    qcf[1:] = x[1:] / -np.expm1(-x[1:])
    total = (45 * si + 7 * sg) / volume
    optical = np.maximum(1 - wn / (1e7 / laser_nm), 0)**4
    # With the harmonic QCF, detailed balance cancels the quantum factor
    # in absorption, leaving beta * omega^2 * S_mu(classical). Do not
    # multiply this channel by the Raman Stokes QCF a second time.
    data = dict(wavenumber_cm1=wn, frequency_THz=freq / 1e12,
                raman_parallel=(45 * si + 4 * sg) / volume,
                raman_crossed=3 * sg / volume, raman_classical=total,
                raman_harmonic_qcf=qcf, raman_stokes=total * qcf * optical)
    if dipole is not None:
        _, smu = welch(dipole, **kw)
        ir_xyz = (2 * np.pi * freq[:, None])**2 * smu / (temperature * volume)
        data.update(ir_x=ir_xyz[:, 0], ir_y=ir_xyz[:, 1], ir_z=ir_xyz[:, 2],
                    ir_absorption=ir_xyz.mean(axis=1))
    return data


def analyze(labels, output, temperature, discard_ps=0., segment_ps=20., laser_nm=532.1, raman_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(labels, allow_pickle=False) as z:
        if not raman_only and ('dipole_convention' not in z or str(z['dipole_convention']) != 'neutral-whole-molecule-v1'):
            raise ValueError('Labels lack a supported periodic dipole convention; raw dipoles cannot be used for IR')
        times, alpha = z['time_ps'], z['polarizability']
        dipole = None if raman_only else z['dipole']
        volume = float(z['volume_A3'])
    if len(times) < 8 or not np.isfinite(times).all() or not np.isfinite(discard_ps) or discard_ps < 0:
        raise ValueError('Invalid time series or equilibration cut')
    dt = float(np.median(np.diff(times)) * 1000)
    if dt <= 0 or not np.allclose(np.diff(times) * 1000, dt, rtol=0, atol=1e-7):
        raise ValueError('Times must increase uniformly')
    keep = times >= discard_ps - 1e-10
    data = calculate(alpha[keep], None if raman_only else dipole[keep], dt, temperature, volume, segment_ps, laser_nm)
    keys = list(data)
    np.savetxt(output / 'spectra.csv', np.column_stack(list(data.values())),
               delimiter=',', header=','.join(keys), comments='')
    metadata = dict(labels_sha256=sha256(labels), temperature_K=temperature, volume_A3=volume,
                    dt_fs=dt, frames=int(keep.sum()), time_ps=[float(times[keep][0]), float(times[keep][-1])],
                    segment_ps=segment_ps, bin_spacing_cm1=float(data['wavenumber_cm1'][1]),
                    nyquist_cm1=1e15 / (2 * dt * C_CM_S), laser_nm=laser_nm,
                    window='Hann, constant detrending, 50% overlap; one-sided PSD per Hz',
                    intensity='Relative response per volume; no absolute cross section or refractive index',
                    raman_units='(e Angstrom^2/V)^2 / (Hz Angstrom^3)',
                    ir_units='(e Angstrom)^2 / (s K Angstrom^3); constant physical prefactors omitted',
                    ir_convention='mean_xyz[omega^2 PSD(mu_xyz)] / (T V); classical/HQCF absorption proxy',
                    dipole_convention='neutral-whole-molecule-v1')
    metadata['spectroscopy'] = 'raman' if raman_only else 'raman-ir'
    if raman_only:
        for key in ['ir_units', 'ir_convention', 'dipole_convention']:
            metadata.pop(key)
    # Independent non-overlapping blocks quantify sampling variation when long enough.
    block_paths = []
    for index in range(1, 4):
        (output / f'block-{index}.csv').unlink(missing_ok=True)
    for index, ids in enumerate(np.array_split(np.flatnonzero(keep), 3)):
        if len(ids) >= round(segment_ps * 1000 / dt):
            block = calculate(alpha[ids], None if raman_only else dipole[ids], dt, temperature, volume, segment_ps, laser_nm)
            name = f'block-{index + 1}.csv'
            np.savetxt(output / name, np.column_stack(list(block.values())), delimiter=',',
                       header=','.join(keys), comments='')
            block_paths.append(name)
    metadata['sampling_blocks'] = block_paths
    save_json(output / 'spectrum.json', metadata)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    channels = [('raman_stokes', 'Raman · Stokes powder response')]
    if not raman_only:
        channels.append(('ir_absorption', 'THz / IR · molecular absorption proxy'))
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 3.5 * len(channels)), layout='constrained', squeeze=False)
    mask = data['wavenumber_cm1'] <= min(4000, metadata['nyquist_cm1'])
    for ax, (key, title) in zip(axes[:, 0], channels):
        curve = data[key]
        norm = max(float(curve[mask].max()), np.finfo(float).tiny)
        ax.plot(data['wavenumber_cm1'][mask], curve[mask] / norm, lw=1)
        ax.set(xlabel='Wavenumber (cm⁻¹)', ylabel='Normalized intensity', title=title)
        ax.grid(alpha=.2)
    fig.savefig(output / 'spectra.png', dpi=160)
    plt.close(fig)
    return metadata
