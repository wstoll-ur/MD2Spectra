"""MACE-MDP labeling, with an explicit neutral whole-molecule IR convention."""
import json
import os
from pathlib import Path
import urllib.request
import numpy as np
from ase.io.trajectory import Trajectory
from .geometry import molecular_geometry, molecular_dipole, validate_frame
from .io import lock, save_json, save_npz, sha256, versions

MODEL_SHA256 = '126f8d1602549e6fa0df775c701a5119ddeb0e3738202af8e7aa736de6c2b692'
MODEL_URL = ('https://raw.githubusercontent.com/Nilsgoe/MACE-MDP/'
             '91ca3140b25a272aad79a8baece2563c166fd766/models/MACE-MDP.model')


def resolve_model(model, cache):
    if model != 'auto':
        path = Path(model).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'MACE-MDP.model'
    with lock(cache / 'download.lock'):
        if not path.exists():
            tmp = path.with_suffix('.download')
            try:
                with urllib.request.urlopen(MODEL_URL, timeout=120) as source, tmp.open('wb') as dest:
                    import shutil
                    shutil.copyfileobj(source, dest)
                if sha256(tmp) != MODEL_SHA256:
                    raise ValueError('Downloaded MACE-MDP checkpoint failed SHA256 verification')
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        if sha256(path) != MODEL_SHA256:
            raise ValueError('Cached MACE-MDP checksum mismatch')
    return path


def response_calculator(model, device):
    # Official checkpoints contain Python model objects; load only trusted files.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(model), device=device,
                          default_dtype='float64', model_type='DipolePolarizabilityMACE')


def predict(calc, atoms, groups, whole):
    alpha = np.asarray(calc.get_property('polarizability', atoms)).copy()
    raw = np.asarray(calc.get_property('dipole', atoms)).copy()
    charges = np.asarray(calc.get_property('charges', atoms)).reshape(-1)
    if alpha.shape != (3, 3) or raw.shape != (3,) or charges.shape != (len(atoms),):
        raise ValueError('Incompatible MACE-MDP response shapes')
    mu = molecular_dipole(raw, charges, atoms.positions, whole, groups)
    if not all(np.isfinite(v).all() for v in [alpha, raw, charges, mu]):
        raise ValueError('Nonfinite response prediction')
    if not np.allclose(alpha, alpha.T, atol=1e-10, rtol=1e-7):
        raise ValueError('Nonsymmetric polarizability')
    return alpha, mu, raw


def qualify(calc, atoms, scale):
    groups, whole, _ = molecular_geometry(atoms, scale)
    alpha, mu, _ = predict(calc, atoms, groups, whole)
    shifted = atoms.copy()
    shifted.positions += [1.234, -2.345, .876]
    shifted.wrap()
    g, w, _ = molecular_geometry(shifted, scale)
    a2, m2, _ = predict(calc, shifted, g, w)
    rotation, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated.set_cell(atoms.cell.array @ rotation.T)
    g, w, _ = molecular_geometry(rotated, scale)
    a3, m3, _ = predict(calc, rotated, g, w)
    np.testing.assert_allclose(a2, alpha, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(m2, mu, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(a3, rotation @ alpha @ rotation.T, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(m3, rotation @ mu, atol=1e-7, rtol=1e-6)
    return dict(passed=True, wrap_alpha_error=float(abs(a2-alpha).max()),
                wrap_dipole_error=float(abs(m2-mu).max()),
                rotation_alpha_error=float(abs(a3-rotation@alpha@rotation.T).max()),
                rotation_dipole_error=float(abs(m3-rotation@mu).max()))


def label(trajectory, output, model='auto', device='cpu', chunk_size=256,
          bond_scale=1.15, cache=None, calculator=None):
    trajectory, output = Path(trajectory), Path(output)
    if chunk_size < 1 or not np.isfinite(bond_scale) or bond_scale <= 0:
        raise ValueError('chunk_size and bond_scale must be positive')
    output.mkdir(parents=True, exist_ok=True)
    cache = cache or Path.home() / '.cache' / 'md2spectra'
    model_path = resolve_model(model, cache) if calculator is None else None
    identity = dict(trajectory_sha256=sha256(trajectory),
                    model_sha256=sha256(model_path) if model_path else 'test-calculator',
                    bond_scale=bond_scale, chunk_size=chunk_size,
                    dipole_convention='neutral-whole-molecule-v1', schema=1)
    with lock(output / 'label.lock'):
        manifest = output / 'labels-manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity:
            raise ValueError('Label input/model/settings changed; use a new label output directory')
        if (output / 'labels-complete.json').exists():
            done = json.loads((output / 'labels-complete.json').read_text())
            if done['identity'] != identity:
                raise ValueError('Completed label provenance mismatch')
            if sha256(output / 'labels.npz') != done['labels_sha256']:
                raise ValueError('Saved labels checksum changed')
            return output / 'labels.npz'
        with Trajectory(trajectory) as frames:
            if len(frames) < 8:
                raise ValueError('At least eight trajectory frames required')
            ref = frames[0]
            validate_frame(ref, ref)
            groups, whole, bonds = molecular_geometry(ref, bond_scale)
            save_json(manifest, dict(identity=identity, runtime=versions(),
                molecule_ids=groups.tolist(), bonds=bonds,
                approximation='Each connected molecule is neutral; no intermolecular charge transfer or ionic conduction'))
            calc = calculator if calculator is not None else response_calculator(model_path, device)
            save_json(output / 'qualification.json', qualify(calc, ref, bond_scale))
            n = len(frames)
            times = np.empty(n)
            alpha, dipoles, raw = np.empty((n, 3, 3)), np.empty((n, 3)), np.empty((n, 3))
            for begin in range(0, n, chunk_size):
                end = min(n, begin + chunk_size)
                path = output / f'chunk-{begin:09d}-{end:09d}.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        if str(z['identity']) != json.dumps(identity, sort_keys=True):
                            raise ValueError('Label chunk provenance mismatch')
                        np.testing.assert_array_equal(z['frame_index'], np.arange(begin, end))
                        times[begin:end], alpha[begin:end] = z['time_ps'], z['polarizability']
                        dipoles[begin:end], raw[begin:end] = z['dipole'], z['raw_dipole']
                else:
                    for k in range(begin, end):
                        atoms = frames[k]
                        validate_frame(atoms, ref)
                        g, w, current_bonds = molecular_geometry(atoms, bond_scale)
                        if current_bonds != bonds or not np.array_equal(g, groups):
                            raise ValueError(f'Molecular connectivity changed at frame {k}; inspect bonding/reactions')
                        times[k] = atoms.info['time_ps']
                        alpha[k], dipoles[k], raw[k] = predict(calc, atoms, g, w)
                    save_npz(path, frame_index=np.arange(begin, end), time_ps=times[begin:end],
                             polarizability=alpha[begin:end], dipole=dipoles[begin:end],
                             raw_dipole=raw[begin:end], identity=json.dumps(identity, sort_keys=True))
                print(f'Response: {end}/{n} frames', flush=True)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or not np.allclose(np.diff(times), np.diff(times)[0], atol=1e-10, rtol=0):
                raise ValueError('Trajectory time_ps metadata must increase uniformly')
            if sha256(trajectory) != identity['trajectory_sha256']:
                raise ValueError('Trajectory changed during labeling')
            save_npz(output / 'labels.npz', time_ps=times, polarizability=alpha, dipole=dipoles,
                     raw_dipole=raw, frame_index=np.arange(n), volume_A3=ref.get_volume(),
                     dipole_convention=identity['dipole_convention'])
            save_json(output / 'labels-complete.json', dict(frames=n,
                        labels_sha256=sha256(output / 'labels.npz'), identity=identity))
    return output / 'labels.npz'
