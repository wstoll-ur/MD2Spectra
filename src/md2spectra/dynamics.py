"""Fixed-cell UMA/Bussi MD with restartable atomic JSON checkpoints."""
import json
import os
import signal
import time
from pathlib import Path
import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from .io import lock, save_json, sha256, versions


def step_count(duration_ps, timestep_fs, name):
    steps = duration_ps * 1000 / timestep_fs
    if not np.isfinite(steps) or abs(steps - round(steps)) > 1e-7:
        raise ValueError(f'{name} must be an exact number of integration steps')
    return round(steps)


def validate_config(cfg):
    for key in ['temperature', 'production_ps', 'timestep_fs', 'thermostat_fs', 'checkpoint_ps', 'fmax']:
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if not np.isfinite(cfg['equilibration_ps']) or cfg['equilibration_ps'] < 0:
        raise ValueError('equilibration_ps must be nonnegative')
    if len(cfg['supercell']) != 3 or any(v < 1 or int(v) != v for v in cfg['supercell']):
        raise ValueError('supercell requires three positive integers')
    if cfg['save_every'] < 1 or cfg['checkpoint_ps'] * 1000 < cfg['save_every'] * cfg['timestep_fs']:
        raise ValueError('Checkpoint interval must be at least one saved-frame interval')
    if cfg['seed'] < 0:
        raise ValueError('seed must be nonnegative')
    equil = step_count(cfg['equilibration_ps'], cfg['timestep_fs'], 'equilibration')
    prod = step_count(cfg['production_ps'], cfg['timestep_fs'], 'production')
    checkpoint = step_count(cfg['checkpoint_ps'], cfg['timestep_fs'], 'checkpoint')
    if any(n % cfg['save_every'] for n in [equil, prod, checkpoint]):
        raise ValueError('Equilibration, production and checkpoint lengths must align with saved frames')
    if prod // cfg['save_every'] < 8:
        raise ValueError('Production requires at least eight saved samples')
    return equil + prod, checkpoint


def prepare(cfg):
    atoms = read(cfg['cif'], index=cfg['cif_index'])
    occupancy = atoms.info.get('occupancy', {})
    if any(len(site) != 1 or not np.isclose(sum(site.values()), 1) for site in occupancy.values()):
        raise ValueError('Partial/mixed CIF occupancies require an ordered structure before MD')
    if not atoms.pbc.all() or atoms.cell.rank != 3 or len(atoms) < 2:
        raise ValueError('Expected a three-dimensional periodic CIF with at least two atoms')
    atoms = atoms.repeat(cfg['supercell'])
    atoms.info = {}
    if not np.isfinite(atoms.positions).all():
        raise ValueError('Invalid CIF coordinates')
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    if distances.min() < .5:
        raise ValueError('Overlapping CIF sites (<0.5 Angstrom); resolve disorder first')
    return atoms


def uma_calculator(cfg):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    settings = InferenceSettings(tf32=False, compile=False)
    if Path(cfg['uma_model']).is_file():
        from fairchem.core.units.mlip_unit import load_predict_unit
        predictor = load_predict_unit(cfg['uma_model'], device=cfg['device'], inference_settings=settings)
    else:
        predictor = pretrained_mlip.get_predict_unit(cfg['uma_model'], device=cfg['device'], inference_settings=settings)
    return FAIRChemCalculator(predictor, task_name=cfg['task'])


def checkpoint(folder, atoms, dyn, rng, step, identity):
    with (folder / 'trajectory.traj').open('rb') as handle:
        os.fsync(handle.fileno())
    save_json(folder / 'checkpoint.json', dict(identity=identity, step=step,
        numbers=atoms.numbers.tolist(), positions=atoms.positions.tolist(),
        momenta=atoms.get_momenta().tolist(), cell=atoms.cell.array.tolist(),
        rng=rng.bit_generator.state, transferred_energy=float(dyn.transferred_energy)))


def recover_trajectory(path, checkpoint_step, save_every):
    """Drop an interrupted tail and require the complete committed prefix."""
    temp = path.with_suffix('.recovered.traj')
    expected = 0
    with Trajectory(path) as old, Trajectory(temp, 'w') as new:
        required = checkpoint_step // save_every + 1
        if len(old) < required:
            raise ValueError('Trajectory is shorter than its committed checkpoint')
        for index in range(required):
            atoms = old[index]
            step = atoms.info['step']
            if step > checkpoint_step:
                break
            if step != expected:
                raise ValueError('Missing/duplicate committed trajectory frame')
            new.write(atoms)
            expected += save_every
    if expected != checkpoint_step + save_every:
        temp.unlink(missing_ok=True)
        raise ValueError('Trajectory does not contain the committed checkpoint frame')
    temp.replace(path)


def run_md(cfg, folder, calculator=None, stop_after_steps=None):
    """calculator/stop_after_steps enable deterministic integration tests."""
    total, checkpoint_steps = validate_config(cfg)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    cfg['cif'] = str(Path(cfg['cif']).resolve())
    identity = dict(config=cfg, cif_sha256=sha256(cfg['cif']), runtime=versions())
    if Path(cfg['uma_model']).is_file():
        identity['uma_sha256'] = sha256(cfg['uma_model'])
    else:
        identity['uma_model_id'] = cfg['uma_model']
        if calculator is None:
            from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
            model_path = pretrained_checkpoint_path_from_name(cfg['uma_model'])
            identity['uma_sha256'] = sha256(model_path)
    with lock(folder / 'md.lock'):
        manifest_path = folder / 'md.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())['identity'] != identity:
                raise ValueError('Run configuration, input or model changed; use a new output directory')
        else:
            save_json(manifest_path, dict(identity=identity, runtime=versions()))
        done = folder / 'md-complete.json'
        traj_path = folder / 'trajectory.traj'
        if done.exists():
            if sha256(traj_path) != json.loads(done.read_text())['trajectory_sha256']:
                raise ValueError('Completed trajectory checksum changed')
            return True
        rng = np.random.default_rng(cfg['seed'])
        cp = folder / 'checkpoint.json'
        saved = json.loads(cp.read_text()) if cp.exists() else None
        if saved:
            if saved['identity'] != identity:
                raise ValueError('Checkpoint provenance mismatch')
            atoms = Atoms(saved['numbers'], positions=saved['positions'], cell=saved['cell'], pbc=True)
            atoms.set_momenta(saved['momenta'])
            rng.bit_generator.state = saved['rng']
            step = saved['step']
        else:
            atoms, step = prepare(cfg), 0
        refcell = atoms.cell.array.copy()
        atoms.calc = calculator if calculator is not None else uma_calculator(cfg)
        if saved is None:
            if cfg['relax']:
                optimizer = FIRE(atoms, logfile=str(folder / 'relax.log'), maxstep=.03)
                if not optimizer.run(fmax=cfg['fmax'], steps=4000):
                    raise RuntimeError('Fixed-cell relaxation did not converge')
            write(folder / 'initial.extxyz', atoms, format='extxyz', write_results=False)
            MaxwellBoltzmannDistribution(atoms, temperature_K=cfg['temperature'], rng=rng)
            Stationary(atoms)
        dyn = Bussi(atoms, cfg['timestep_fs'] * units.fs, temperature_K=cfg['temperature'],
                    taut=cfg['thermostat_fs'] * units.fs, rng=rng)
        if saved:
            dyn.transferred_energy = saved['transferred_energy']
            recover_trajectory(traj_path, step, cfg['save_every'])
        stop = [False]
        previous_handlers = {}
        def request_stop(signum, frame):
            stop[0] = True
        for sig in [signal.SIGINT, signal.SIGTERM]:
            previous_handlers[sig] = signal.signal(sig, request_stop)
        start = time.monotonic()
        try:
            with Trajectory(traj_path, 'a' if saved else 'w') as trajectory:
                def save_frame():
                    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.get_forces()).all():
                        raise RuntimeError('Nonfinite MD geometry/forces')
                    if not np.allclose(atoms.cell, refcell, atol=1e-10, rtol=0):
                        raise RuntimeError('Cell changed during fixed-cell MD')
                    frame = atoms.copy()
                    frame.info = dict(step=step, time_ps=step * cfg['timestep_fs'] / 1000,
                                      temperature_K=float(atoms.get_temperature()))
                    trajectory.write(frame)
                if saved is None:
                    save_frame()
                    checkpoint(folder, atoms, dyn, rng, step, identity)
                while step < total and not stop[0] and not (folder / 'STOP').exists():
                    dyn.run(cfg['save_every'])
                    step += cfg['save_every']
                    save_frame()
                    if step % checkpoint_steps == 0 or step == total:
                        checkpoint(folder, atoms, dyn, rng, step, identity)
                        save_json(folder / 'progress.json', dict(step=step, target_steps=total,
                            time_ps=step * cfg['timestep_fs'] / 1000, elapsed_s=time.monotonic() - start))
                        print(f'MD: {step}/{total} steps', flush=True)
                    if stop_after_steps is not None and step >= stop_after_steps:
                        break
                checkpoint(folder, atoms, dyn, rng, step, identity)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        if step != total:
            return False
        save_json(done, dict(steps=step, frames=step // cfg['save_every'] + 1,
                            trajectory_sha256=sha256(traj_path)))
        return True
