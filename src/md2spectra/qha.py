"""Volume-only QHA using UMA energies and ASE finite-displacement phonons.

All energies/volumes refer to the input cell, not the phonon supercell.
The phonon scan fixes cell shape. Each fitted temperature cell then relaxes
its shape and internal coordinates at exactly the fitted volume before NVT.
"""
import json
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq
from ase import units
from ase.dft.kpoints import monkhorst_pack
from ase.filters import FrechetCellFilter
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read, write
from ase.optimize import FIRE
from ase.phonons import Phonons

from .dynamics import prepare, uma_calculator
from .io import lock, save_json, save_npz, sha256, versions


def validate_qha(cfg):
    temperatures = np.asarray(cfg['temperatures'], dtype=float)
    factors = np.asarray(cfg['volume_factors'], dtype=float)
    if (temperatures.ndim != 1 or not len(temperatures) or
            not np.isfinite(temperatures).all() or np.any(temperatures < 0) or
            np.any(np.diff(temperatures) <= 0)):
        raise ValueError('QHA temperatures must be finite, nonnegative and strictly increasing')
    if (factors.ndim != 1 or len(factors) < 5 or not np.isfinite(factors).all() or
            np.any(factors <= 0) or np.any(np.diff(factors) <= 0)):
        raise ValueError('QHA requires at least five positive, increasing volume factors (V/V_input)')
    for key in ['phonon_supercell', 'mesh']:
        values = cfg[key]
        if len(values) != 3 or any(not np.isfinite(v) or v < 1 or int(v) != v for v in values):
            raise ValueError(f'QHA {key} requires three positive integers')
    # A shifted even MP mesh avoids the singular acoustic modes at Gamma.
    # Gamma is checked separately for instabilities, never included in F_vib.
    if any(v % 2 for v in cfg['mesh']):
        raise ValueError('QHA mesh dimensions must be even (shifted mesh excluding Gamma)')
    for key in ['displacement', 'fmax']:
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'QHA {key} must be positive and finite')
    if not np.isfinite(cfg['pressure_gpa']):
        raise ValueError('QHA pressure must be finite')
    stress_limit = cfg.get('shape_stress_gpa', .01)
    if not np.isfinite(stress_limit) or stress_limit <= 0:
        raise ValueError('QHA shape_stress_gpa must be positive and finite')


def vibrational_free_energy(energies, temperatures):
    """Quantum harmonic F in eV/input cell, with equal q weights and ZPE.

    energies is (nq, 3*natoms), in eV. Negative ASE energies encode imaginary
    frequencies. No modes may be discarded, clipped or made positive by abs().
    """
    energies = np.asarray(energies, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    if (energies.ndim != 2 or not energies.size or not np.isfinite(energies).all()
            or np.any(energies <= 0)):
        raise ValueError('Unstable or zero phonon modes on the QHA mesh; inspect relaxation/supercell convergence')
    if temperatures.ndim != 1 or not np.isfinite(temperatures).all() or np.any(temperatures < 0):
        raise ValueError('Invalid QHA temperatures')
    zpe = .5 * energies.sum(axis=1).mean()
    result = np.full(len(temperatures), zpe)
    for i, temperature in enumerate(temperatures):
        if temperature > 0:
            kbt = units.kB * temperature
            result[i] += kbt * np.log(-np.expm1(-energies / kbt)).sum(axis=1).mean()
    return result


def fit_equilibrium(volumes, free_energy):
    """Cubic fit with a bracketed, unique, convex minimum; never extrapolate."""
    volumes, free_energy = np.asarray(volumes), np.asarray(free_energy)
    if (volumes.ndim != 1 or len(volumes) < 5 or free_energy.shape != volumes.shape
            or not np.isfinite(volumes).all() or not np.isfinite(free_energy).all()
            or np.any(volumes <= 0) or np.any(np.diff(volumes) <= 0)):
        raise ValueError('Invalid QHA volume/free-energy grid')
    index = int(np.argmin(free_energy))
    if index in [0, len(volumes) - 1]:
        raise ValueError('QHA minimum is not bracketed; expand the volume-factor range in a new output directory')
    # Remove the large static energy offset before fitting small differences.
    offset = float(free_energy.min())
    poly = Polynomial.fit(volumes, free_energy - offset, 3)
    curvature = poly.deriv(2)
    if np.any(curvature(volumes[[0, -1]]) <= 0):
        raise ValueError('Nonconvex QHA fit; inspect the free-energy grid and refine the volume range')
    derivative = poly.deriv()
    if derivative(volumes[0]) >= 0 or derivative(volumes[-1]) <= 0:
        raise ValueError('QHA fitted minimum lies outside the sampled volume range')
    # Bracketing avoids the ill-conditioned quadratic roots when the cubic
    # coefficient is nearly zero (e.g. an exactly parabolic free-energy curve).
    volume = float(brentq(derivative, volumes[0], volumes[-1], xtol=1e-12))
    if not volumes[index-1] < volume < volumes[index+1]:
        raise ValueError('QHA fitted minimum is not reliably bracketed by sampled volumes')
    residual = poly(volumes) + offset - free_energy
    return dict(volume_A3=volume, free_energy_eV=float(poly(volume) + offset),
                bulk_modulus_GPa=float(volume * curvature(volume) / units.GPa),
                fit_rms_eV=float(np.sqrt(np.mean(residual**2))),
                fit_max_error_eV=float(np.max(abs(residual))))


def _relax(atoms, calculator, folder, fmax):
    """Persist the exact phonon reference so a restart cannot mix geometries."""
    folder.mkdir(parents=True, exist_ok=True)
    path, manifest = folder / 'relaxed.extxyz', folder / 'relaxed.json'
    if manifest.exists():
        if sha256(path) != json.loads(manifest.read_text())['sha256']:
            raise ValueError(f'QHA relaxed structure checksum changed: {path}')
        return read(path)
    atoms.calc = calculator
    optimizer = FIRE(atoms, logfile=str(folder / 'relax.log'), maxstep=.03)
    if not optimizer.run(fmax=fmax, steps=4000):
        raise RuntimeError(f'QHA fixed-cell relaxation did not converge: {folder}')
    energy = float(atoms.get_potential_energy())
    if not np.isfinite(energy):
        raise ValueError('Nonfinite QHA static energy')
    tmp = folder / 'relaxed.tmp.extxyz'
    write(tmp, atoms, format='extxyz', write_results=False)
    tmp.replace(path)
    save_json(manifest, dict(sha256=sha256(path), energy_eV=energy))
    return atoms


class FixedVolumeCellFilter(FrechetCellFilter):
    """Constrain log(det F) = trace(log F) = 0 on every optimizer update.

    ASE projects cell forces with constant_volume=True. Explicitly projecting
    the log-cell coordinates also keeps finite updates on the volume manifold.
    All deviatoric strains, including shear, remain free.
    """
    def __init__(self, atoms):
        super().__init__(atoms, constant_volume=True)
        # ASE versions differ in whether log-cell coordinates include the
        # atom-count scale inside the logarithm. Preserve the reference trace
        # so the *relative* log determinant is zero in either convention.
        self.reference_trace = float(np.trace(super().get_positions()[-3:]))

    def set_positions(self, new, **kwargs):
        new = np.array(new, copy=True)
        new[-3:] -= np.eye(3) * (np.trace(new[-3:]) - self.reference_trace) / 3
        super().set_positions(new, **kwargs)


def _shape_metrics(atoms, target_volume):
    volume = atoms.get_volume()
    if not np.isfinite(volume) or not np.isclose(volume, target_volume, rtol=1e-10, atol=0):
        raise RuntimeError('Constant-volume cell optimization changed the QHA volume')
    forces = atoms.get_forces()
    try:
        stress = atoms.get_stress(voigt=False)
    except PropertyNotImplementedError as error:
        raise RuntimeError('QHA cell-shape optimization requires a calculator with stress support') from error
    if not np.isfinite(forces).all() or not np.isfinite(stress).all():
        raise ValueError('Nonfinite forces/stress during QHA cell-shape optimization')
    deviatoric = stress - np.eye(3) * np.trace(stress) / 3
    return dict(volume_A3=volume, max_force_eV_A=float(np.linalg.norm(forces, axis=1).max()),
                max_deviatoric_stress_GPa=float(np.abs(deviatoric).max() / units.GPa))


def _relax_shape(atoms, calculator, folder, target_volume, fmax, stress_gpa):
    """Minimize static energy at fixed V(T), with independent force/stress checks."""
    folder.mkdir(parents=True, exist_ok=True)
    path, manifest = folder / 'relaxed.extxyz', folder / 'relaxed.json'
    settings = dict(method='fixed-volume-log-cell-v1', target_volume_A3=target_volume,
                    fmax_eV_A=fmax, deviatoric_stress_limit_GPa=stress_gpa)
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if record.get('settings') != settings or sha256(path) != record['sha256']:
            raise ValueError(f'QHA shape-relaxation settings/checksum changed: {path}')
        saved = read(path)
        if not np.isclose(saved.get_volume(), target_volume, rtol=1e-10, atol=0):
            raise ValueError('Saved QHA cell volume differs from its fitted volume')
        return saved
    atoms.set_cell(atoms.cell * (target_volume/atoms.get_volume())**(1/3), scale_atoms=True)
    atoms.calc = calculator
    metrics = _shape_metrics(atoms, target_volume)
    initial_energy = float(atoms.get_potential_energy())
    cell_filter = FixedVolumeCellFilter(atoms)
    converged = False
    # Use physical force and deviatoric-stress tolerances, independent of the
    # arbitrary atom-count scaling of the filter's cell degrees of freedom.
    with FIRE(cell_filter, logfile=str(folder / 'relax.log'), maxstep=.03) as optimizer:
        for _ in optimizer.irun(fmax=0., steps=4000):
            metrics = _shape_metrics(atoms, target_volume)
            if (metrics['max_force_eV_A'] <= fmax and
                    metrics['max_deviatoric_stress_GPa'] <= stress_gpa):
                converged = True
                break
    if not converged:
        raise RuntimeError(f'QHA constant-volume shape relaxation did not converge: {folder}')
    energy = float(atoms.get_potential_energy())
    if not np.isfinite([initial_energy, energy]).all():
        raise ValueError('Nonfinite QHA shape-relaxation energy')
    tmp = folder / 'relaxed.tmp.extxyz'
    write(tmp, atoms, format='extxyz', write_results=False)
    tmp.replace(path)
    save_json(manifest, dict(settings=settings, sha256=sha256(path), energy_eV=energy,
                            initial_energy_eV=initial_energy, converged=True, **metrics,
                            cell_A=atoms.cell.array.tolist(),
                            lengths_A=atoms.cell.lengths().tolist(), angles_deg=atoms.cell.angles().tolist()))
    return atoms


def _phonopy_from_cache(phonons):
    """Read ASE displacements with explicit atom mapping; interpolate in Phonopy.

    The pair-dependent shortest supercell vectors matter for small molecular
    supercells. A single lattice translation per force-constant block can give
    spurious imaginary modes away from commensurate q points.
    """
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError as error:
        raise RuntimeError('QHA requires Phonopy; install md2spectra[qha]') from error
    atoms = phonons.atoms
    model = Phonopy(PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                    cell=atoms.cell.array, scaled_positions=atoms.get_scaled_positions(),
                    masses=atoms.get_masses()), phonons.supercell,
                    primitive_matrix=np.eye(3), is_symmetry=False)
    # ASE orders repeated cells first, whereas Phonopy orders atoms first.
    repeated = atoms.repeat(phonons.supercell)
    delta = model.supercell.scaled_positions[:, None] - repeated.get_scaled_positions()[None, :]
    delta -= np.rint(delta)
    distances = np.linalg.norm(delta @ repeated.cell.array, axis=2)
    p2a = distances.argmin(axis=1)
    if (len(np.unique(p2a)) != len(repeated) or
            np.max(distances[np.arange(len(repeated)), p2a]) > 1e-7 or
            not np.array_equal(model.supercell.numbers, repeated.numbers[p2a])):
        raise ValueError('Cannot uniquely map ASE and Phonopy supercell atoms')
    a2p = np.argsort(p2a)
    displacements = []
    for atom in range(len(atoms)):
        for axis, letter in enumerate('xyz'):
            for sign, suffix in [(-1, '-'), (1, '+')]:
                forces = np.asarray(phonons.cache[f'{atom}{letter}{suffix}']['forces'])
                if forces.shape != (len(repeated), 3) or not np.isfinite(forces).all():
                    raise ValueError('Invalid cached phonon forces')
                displacements.append(dict(number=int(a2p[atom]),
                    displacement=np.eye(3)[axis] * sign * phonons.delta,
                    forces=forces[p2a]))
    model.dataset = dict(natom=len(repeated), first_atoms=displacements)
    model.produce_force_constants(fc_calculator='traditional')
    # Enforce translational and permutation identities on the full tensor.
    # No frequency is clipped, discarded, or replaced by its absolute value.
    model.symmetrize_force_constants()
    return model


def _phonons(atoms, calculator, folder, cfg):
    phonons = Phonons(atoms, calculator, supercell=tuple(cfg['phonon_supercell']),
                      delta=cfg['displacement'], name=str(folder / 'forces'))
    # ASE leaves an empty cache entry if killed during a force calculation.
    # The outer QHA lock ensures no other writer owns these placeholders.
    for path in (folder / 'forces').glob('cache.*.json'):
        if path.stat().st_size == 0:
            path.unlink()
    phonons.run()
    model = _phonopy_from_cache(phonons)
    from phonopy.physical_units import get_physical_units
    qpoints = np.concatenate((np.zeros((1, 3)), monkhorst_pack(cfg['mesh'])))
    frequencies = model.run_qpoints(qpoints).frequencies
    all_energies = frequencies * get_physical_units().THzToEv
    gamma, energies = all_energies[0], all_energies[1:]
    # Retain the actual rejected spectrum so scientific failures are diagnosable.
    save_npz(folder / 'phonon-diagnostics.npz', qpoints=qpoints, energies_eV=all_energies)
    save_json(folder / 'phonon-diagnostics.json', dict(backend='phonopy', version=version('phonopy'),
        gamma_eV=gamma.tolist(), mesh_min_eV=float(energies.min()),
        nonpositive_mesh_modes=int(np.sum(energies <= 0))))
    if (not np.isfinite(gamma).all() or np.any(abs(gamma[:3]) > 1e-4)
            or np.any(gamma[3:] <= 0)):
        raise ValueError(f'Unstable Gamma phonons or acoustic sum-rule failure; inspect {folder / "phonon-diagnostics.json"}')
    # Validate before saving; an unstable volume must never be silently dropped.
    vibrational_free_energy(energies, [0.])
    return energies


def run_qha(cfg, output, calculator=None):
    """Restartable volume scan and temperature cells; calculator is a test seam."""
    cfg = dict(cfg)
    # QHA alternates input-cell relaxations and phonon supercells. Do not use
    # the composition-specialized MD inference profile across those sizes.
    cfg['uma_inference'] = 'conservative'
    cfg.setdefault('shape_stress_gpa', .01)
    validate_qha(cfg)
    cfg['cif'] = str(Path(cfg['cif']).expanduser().resolve())
    if Path(cfg['uma_model']).expanduser().is_file():
        cfg['uma_model'] = str(Path(cfg['uma_model']).expanduser().resolve())
    reference = prepare(dict(cfg, supercell=[1, 1, 1]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = versions()
    try:
        runtime['phonopy'] = version('phonopy')
    except PackageNotFoundError as error:
        raise RuntimeError('QHA requires Phonopy; install md2spectra[qha]') from error
    identity = dict(schema=3, config=cfg, cif_sha256=sha256(cfg['cif']), runtime=runtime,
                    method='isotropic-phonopy-cubic-shape-relax-v3')
    if calculator is not None:
        identity['model_sha256'] = 'test-calculator'
    else:
        model_path = Path(cfg['uma_model'])
        if not model_path.is_file():
            from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
            model_path = pretrained_checkpoint_path_from_name(cfg['uma_model'])
        identity['model_sha256'] = sha256(model_path)
    with lock(output / 'qha.lock'):
        manifest, done = output / 'qha.json', output / 'qha-complete.json'
        if manifest.exists():
            if json.loads(manifest.read_text())['identity'] != identity:
                raise ValueError('QHA input/model/settings changed; use a new output directory')
        else:
            save_json(manifest, dict(identity=identity))
        if done.exists():
            result = json.loads(done.read_text())
            if result['identity'] != identity:
                raise ValueError('Completed QHA provenance mismatch')
            for name, checksum in result['files'].items():
                if sha256(output / name) != checksum:
                    raise ValueError(f'Completed QHA checksum changed: {name}')
            return result
        calc = calculator if calculator is not None else uma_calculator(cfg)
        volumes, static, vibrations, relaxed = [], [], [], []
        files = {}
        for index, factor in enumerate(cfg['volume_factors']):
            folder = output / f'volume-{index:03d}'
            atoms = reference.copy()
            atoms.set_cell(reference.cell * factor**(1/3), scale_atoms=True)
            atoms = _relax(atoms, calc, folder, cfg['fmax'])
            relaxed.append(atoms)
            volume_done = folder / 'complete.json'
            data = folder / 'phonons.npz'
            if volume_done.exists():
                record = json.loads(volume_done.read_text())
                if sha256(data) != record['sha256']:
                    raise ValueError(f'QHA phonon checksum changed: {data}')
                with np.load(data, allow_pickle=False) as saved:
                    energies = saved['energies_eV']
            else:
                energies = _phonons(atoms, calc, folder, cfg)
                save_npz(data, energies_eV=energies)
                save_json(volume_done, dict(sha256=sha256(data)))
            expected = (int(np.prod(cfg['mesh'])), 3 * len(reference))
            if energies.shape != expected:
                raise ValueError(f'Unexpected phonon array shape: {energies.shape}, expected {expected}')
            volumes.append(atoms.get_volume())
            static.append(json.loads((folder / 'relaxed.json').read_text())['energy_eV'])
            vibrations.append(vibrational_free_energy(energies, cfg['temperatures']))
            for path in [data, folder / 'relaxed.extxyz', folder / 'relaxed.json', volume_done]:
                files[str(path.relative_to(output))] = sha256(path)
            print(f'QHA: volume {index+1}/{len(cfg["volume_factors"])} complete', flush=True)
        volumes, static, vibrations = map(np.asarray, [volumes, static, vibrations])
        free_energy = static[:, None] + vibrations + cfg['pressure_gpa'] * units.GPa * volumes[:, None]
        # Fit every temperature before starting any temperature-dependent MD.
        rows = [dict(temperature_K=float(t), **fit_equilibrium(volumes, free_energy[:, i]))
                for i, t in enumerate(cfg['temperatures'])]
        save_npz(output / 'free-energy.npz', volumes_A3=volumes, static_energy_eV=static,
                 temperatures_K=cfg['temperatures'], vibrational_free_energy_eV=vibrations,
                 total_free_energy_eV=free_energy)
        for index, row in enumerate(rows):
            nearest = int(np.argmin(abs(volumes - row['volume_A3'])))
            atoms = relaxed[nearest].copy()
            atoms.set_cell(reference.cell * (row['volume_A3']/reference.get_volume())**(1/3), scale_atoms=True)
            folder = output / 'cells' / f'temperature-{index:03d}'
            atoms = _relax_shape(atoms, calc, folder, row['volume_A3'],
                                 cfg['fmax'], cfg['shape_stress_gpa'])
            row['structure'] = str((folder / 'relaxed.extxyz').relative_to(output))
            row['structure_sha256'] = sha256(output / row['structure'])
            row['cell_A'] = atoms.cell.array.tolist()
            row['lengths_A'] = atoms.cell.lengths().tolist()
            row['angles_deg'] = atoms.cell.angles().tolist()
            files[row['structure']] = row['structure_sha256']
            files[str((folder / 'relaxed.json').relative_to(output))] = sha256(folder / 'relaxed.json')
        np.savetxt(output / 'volumes.csv',
                   [[r['temperature_K'], r['volume_A3'], r['bulk_modulus_GPa'], r['fit_rms_eV']] for r in rows],
                   delimiter=',', header='temperature_K,volume_A3,bulk_modulus_GPa,fit_rms_eV', comments='')
        for name in ['free-energy.npz', 'volumes.csv']:
            files[name] = sha256(output / name)
        if sha256(cfg['cif']) != identity['cif_sha256']:
            raise ValueError('QHA input changed during calculation')
        result = dict(identity=identity, reference_volume_A3=reference.get_volume(),
                      cell_model='Isotropic QHA volume scan; static fixed-volume shape/position relaxation at each temperature',
                      temperatures=rows, files=files)
        save_json(done, result)
        return result
