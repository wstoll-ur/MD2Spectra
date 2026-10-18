"""Two-stage model execution keeps UMA and MACE dtype/device state isolated."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .io import lock, save_json, sha256


def add_source_options(p):
    p.add_argument('--cif', required=True)
    p.add_argument('--cif-index', type=int, default=0, help='Zero-based structure index in a multi-block CIF')
    p.add_argument('--uma-model', default='uma-s-1p2p1', help='FAIRChem model ID or trusted local checkpoint')
    p.add_argument('--uma-inference', choices=['conservative', 'md'], default=None,
                   help='MD profile (default conservative); md merges experts; QHA always uses conservative inference')
    p.add_argument('--task', default='omc', choices=['omc', 'omat', 'omol', 'oc20', 'oc22', 'oc25', 'odac'])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--output', type=Path, required=True)


def add_md_options(p, multiple_temperatures=False):
    add_source_options(p)
    temperatures = p.add_mutually_exclusive_group(required=True)
    temperatures.add_argument('--temperature', type=float, help='Kelvin')
    if multiple_temperatures:
        temperatures.add_argument('--temperatures', type=float, nargs='+',
                                  help='Increasing temperatures in Kelvin; requires --qha')
        temperatures.add_argument('--temperature-range', type=float, nargs=3,
                                  metavar=('START', 'STOP', 'STEP'),
                                  help='Inclusive Kelvin range; STOP must align with STEP; requires --qha')
    p.add_argument('--supercell', type=int, nargs=3, default=[1, 1, 1], metavar=('NX', 'NY', 'NZ'))
    p.add_argument('--production-ps', type=float, default=100.)
    p.add_argument('--equilibration-ps', type=float, default=10.)
    p.add_argument('--timestep-fs', type=float, default=.25)
    p.add_argument('--save-every', type=int, default=4, help='Save every N integration steps')
    p.add_argument('--save-momenta', action=argparse.BooleanOptionalAction, default=True,
                   help='Store momenta in every frame; checkpoint momenta are always retained')
    p.add_argument('--thermostat-fs', type=float, default=100.)
    p.add_argument('--checkpoint-ps', type=float, default=1.)
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--relax', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fmax', type=float, default=.02, help='Fixed-cell relaxation force threshold, eV/Angstrom')


def add_qha_options(p):
    p.add_argument('--qha-relax-reference', action=argparse.BooleanOptionalAction, default=False,
                   help='Fully relax input-cell positions, lengths and angles before the volume scan')
    p.add_argument('--qha-volume-factors', type=float, nargs='+',
                   default=[.94, .96, .98, 1., 1.02, 1.04, 1.06],
                   help='At least five increasing V/V_reference values; reference is optimized when requested')
    p.add_argument('--qha-supercell', type=int, nargs=3, default=[2, 2, 2],
                   help='Finite-displacement supercell, independent of the MD supercell')
    p.add_argument('--qha-mesh', type=int, nargs=3, default=[8, 8, 8],
                   help='Even dimensions of a shifted phonon q-point mesh')
    p.add_argument('--qha-displacement', type=float, default=.01, help='Phonon displacement in Angstrom')
    p.add_argument('--qha-fmax', type=float, default=.001, help='QHA internal relaxation threshold, eV/Angstrom')
    p.add_argument('--qha-shape-stress-gpa', type=float, default=.01,
                   help='Deviatoric stress tolerance for fixed-volume temperature-cell optimization, GPa')
    p.add_argument('--qha-pressure-gpa', type=float, default=0., help='External pressure in GPa')


def qha_config(args, temperatures):
    cfg = {key: getattr(args, key) for key in ['cif', 'cif_index', 'uma_model', 'task', 'device']}
    cfg['cif'] = str(Path(cfg['cif']).expanduser().resolve())
    if Path(cfg['uma_model']).expanduser().is_file():
        cfg['uma_model'] = str(Path(cfg['uma_model']).expanduser().resolve())
    cfg['uma_inference'] = 'conservative'
    cfg.update(temperatures=temperatures, volume_factors=args.qha_volume_factors,
               phonon_supercell=args.qha_supercell, mesh=args.qha_mesh,
               displacement=args.qha_displacement, fmax=args.qha_fmax,
               pressure_gpa=args.qha_pressure_gpa, shape_stress_gpa=args.qha_shape_stress_gpa,
               relax_reference=args.qha_relax_reference)
    from .qha import validate_qha
    validate_qha(cfg)
    return cfg


def temperature_values(args):
    if args.temperature_range is None:
        return args.temperatures
    import numpy as np
    start, stop, step = args.temperature_range
    if not np.isfinite([start, stop, step]).all() or start < 0 or stop < start or step <= 0:
        raise ValueError('Temperature range needs 0 <= START <= STOP and STEP > 0')
    intervals = (stop - start) / step
    if not np.isfinite(intervals) or intervals > 10000:
        raise ValueError('Temperature range exceeds 10001 temperatures')
    if not np.isclose(intervals, round(intervals), rtol=0, atol=1e-8):
        raise ValueError('Temperature range STOP must align with START + N * STEP')
    return np.linspace(start, stop, round(intervals) + 1).tolist()


def command_options(values):
    options = []
    for key, value in values.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            options.append(flag if value else '--no-' + key.replace('_', '-'))
        elif isinstance(value, list):
            options += [flag, *map(str, value)]
        else:
            options += [flag, str(value)]
    return options


def run_qha_series(args, cfg, qcfg):
    """Separate QHA/MD/response processes release model memory between stages."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis = dict(mdp_model=args.mdp_model, response_python=args.response_python,
                    response_device=args.response_device or args.device, segment_ps=args.segment_ps,
                    laser_nm=args.laser_nm, bond_scale=args.bond_scale, max_wavenumber=args.max_wavenumber)
    identity = dict(md=cfg, qha=qcfg, analysis=analysis, raman_only=args.raman_only,
                    cif_sha256=sha256(cfg['cif']))
    with lock(output / 'series.lock'):
        if (output / 'md.json').exists():
            raise ValueError('This directory already contains a fixed-cell run; choose a new QHA output directory')
        manifest = output / 'series.json'
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError('Temperature-series settings changed; use a new output directory')
        save_json(manifest, identity)
        qoptions = {key: qcfg[key] for key in ['cif', 'cif_index', 'uma_model', 'task', 'device', 'temperatures']}
        if 'uma_inference' in qcfg:
            qoptions['uma_inference'] = qcfg['uma_inference']
        qoptions.update({key: value for key, value in vars(args).items() if key.startswith('qha_')})
        subprocess.run([sys.executable, '-m', 'md2spectra', 'qha', '--output', str(output / 'qha')]
                       + command_options(qoptions), check=True)
        result = json.loads((output / 'qha' / 'qha-complete.json').read_text())
        runs = []
        for index, row in enumerate(result['temperatures']):
            structure = output / 'qha' / row['structure']
            if sha256(structure) != row['structure_sha256']:
                raise ValueError('QHA temperature cell checksum changed')
            folder = output / f'temperature-{index:03d}-{row["temperature_K"]:g}K'
            child = dict(cfg, cif=str(structure), cif_index=0, temperature=row['temperature_K'])
            subprocess.run([sys.executable, '-m', 'md2spectra', 'run', '--output', str(folder)]
                           + command_options(child) + command_options(analysis)
                           + (['--raman-only'] if args.raman_only else []), check=True)
            runs.append(dict(temperature_K=row['temperature_K'], unit_cell_volume_A3=row['volume_A3'],
                             directory=str(folder.relative_to(output))))
        save_json(output / 'complete.json', dict(status='completed', runs=runs,
                  qha_sha256=sha256(output / 'qha' / 'qha-complete.json')))
    print(f'Completed QHA temperature series: {output}')


MD_KEYS = ['cif', 'cif_index', 'temperature', 'supercell', 'production_ps', 'equilibration_ps',
           'timestep_fs', 'save_every', 'thermostat_fs', 'checkpoint_ps', 'seed', 'uma_model',
           'task', 'device', 'relax', 'fmax', 'save_momenta', 'uma_inference']


def main(argv=None):
    parser = argparse.ArgumentParser(description='CIF → UMA MD → molecular Raman and THz/IR spectra')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Run the complete restartable workflow')
    add_md_options(run, multiple_temperatures=True)
    run.add_argument('--qha', action='store_true', help='Fit QHA volumes before running MD/spectra at each temperature')
    add_qha_options(run)
    run.add_argument('--mdp-model', default='auto')
    run.add_argument('--raman-only', action='store_true', help='Compute polarizabilities and Raman only; skip dipole/IR analysis')
    run.add_argument('--response-python', default=sys.executable, help='Python in an optional separate MACE environment')
    run.add_argument('--response-device', choices=['cpu', 'cuda'], default=None)
    run.add_argument('--segment-ps', type=float, default=20.)
    run.add_argument('--laser-nm', type=float, default=532.1)
    run.add_argument('--bond-scale', type=float, default=1.15)
    run.add_argument('--max-wavenumber', type=float, default=4000., help='Required spectral coverage for the sampling check')
    run.add_argument('--dry-run', action='store_true', help='Validate CIF/configuration without loading models or starting MD')
    md = sub.add_parser('md', help='Only run UMA dynamics')
    add_md_options(md)
    md.add_argument('--max-wall-s', type=float, help='Checkpoint and stop after this wall time; rerun to continue')
    qha = sub.add_parser('qha', help='Only calculate QHA volumes and relaxed temperature-dependent cells')
    add_source_options(qha)
    add_qha_options(qha)
    qha_temperatures = qha.add_mutually_exclusive_group(required=True)
    qha_temperatures.add_argument('--temperatures', type=float, nargs='+', help='Increasing Kelvin values; 0 K is allowed')
    qha_temperatures.add_argument('--temperature-range', type=float, nargs=3, metavar=('START', 'STOP', 'STEP'),
                                 help='Inclusive Kelvin range; STOP must align with STEP')
    qha.add_argument('--dry-run', action='store_true', help='Validate and preview the scan without loading UMA')
    label = sub.add_parser('label', help='Label a completed ASE trajectory with time_ps metadata')
    label.add_argument('--trajectory', type=Path, required=True)
    label.add_argument('--output', type=Path, required=True)
    label.add_argument('--mdp-model', default='auto')
    label.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    label.add_argument('--chunk-size', type=int, default=256)
    label.add_argument('--bond-scale', type=float, default=1.15)
    label.add_argument('--raman-only', action='store_true')
    label.add_argument('--max-wall-s', type=float, help='Raman-only: checkpoint chunks and stop at this wall time')
    spec = sub.add_parser('spectrum', help='Recalculate spectra from response labels')
    spec.add_argument('--labels', type=Path, required=True)
    spec.add_argument('--output', type=Path, required=True)
    spec.add_argument('--temperature', type=float, required=True)
    spec.add_argument('--discard-ps', type=float, default=0.)
    spec.add_argument('--segment-ps', type=float, default=20.)
    spec.add_argument('--laser-nm', type=float, default=532.1)
    spec.add_argument('--raman-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'qha':
            from .dynamics import prepare
            from .qha import run_qha
            cfg = qha_config(args, temperature_values(args))
            atoms = prepare(dict(cfg, supercell=[1, 1, 1]))
            if args.dry_run:
                print(json.dumps(dict(config=cfg, input_cell_atoms=len(atoms),
                      input_cell_volume_A3=atoms.get_volume(),
                      volume_model='Isotropic QHA scan, then fixed-volume shape/position optimization at each temperature'), indent=2))
            else:
                run_qha(cfg, args.output)
        elif args.command == 'label':
            from .response import label as label_trajectory
            if args.raman_only:
                from .response import label_raman
                completed = label_raman(args.trajectory, args.output, args.mdp_model, args.device,
                                        args.chunk_size, max_wall_s=args.max_wall_s)
                if completed is None:
                    parser.exit(75, 'Raman labels checkpointed before completion; rerun to continue.\n')
            else:
                if args.max_wall_s is not None:
                    raise ValueError('--max-wall-s for labeling currently requires --raman-only')
                label_trajectory(args.trajectory, args.output, args.mdp_model, args.device,
                                 args.chunk_size, args.bond_scale)
        elif args.command == 'spectrum':
            from .spectra import analyze
            analyze(args.labels, args.output, args.temperature, args.discard_ps, args.segment_ps, args.laser_nm, args.raman_only)
        else:
            from .dynamics import validate_config, prepare, run_md
            qcfg = None
            if args.command == 'run':
                selected_temperatures = temperature_values(args)
                if selected_temperatures is not None and not args.qha:
                    raise ValueError('--temperatures/--temperature-range requires --qha')
                if args.qha:
                    temperatures = selected_temperatures if selected_temperatures is not None else [args.temperature]
                    qcfg = qha_config(args, temperatures)
                    if any(t <= 0 for t in temperatures):
                        raise ValueError('MD/spectra temperatures must be positive; use qha alone for 0 K')
                    args.temperature = temperatures[0]
            cfg = {key: getattr(args, key) for key in MD_KEYS}
            if cfg['uma_inference'] is None:
                del cfg['uma_inference']  # Preserve identities of runs made before this option.
            cfg['cif'] = str(Path(cfg['cif']).expanduser().resolve())
            if Path(cfg['uma_model']).expanduser().is_file():
                cfg['uma_model'] = str(Path(cfg['uma_model']).expanduser().resolve())
            validate_config(cfg)
            if args.command == 'md':
                if not run_md(cfg, args.output, max_wall_s=args.max_wall_s):
                    parser.exit(75, 'MD checkpointed before completion. Remove STOP if present and rerun the same command.\n')
                return
            import numpy as np
            from .spectra import C_CM_S
            nyquist = 1e15 / (2 * args.timestep_fs * args.save_every * C_CM_S)
            if not np.isfinite(args.max_wavenumber) or args.max_wavenumber <= 0 or args.max_wavenumber >= nyquist:
                raise ValueError(f'Required spectral coverage must be below Nyquist ({nyquist:.1f} cm^-1); save more frequently')
            if not np.isfinite(args.segment_ps) or not 0 < args.segment_ps <= args.production_ps:
                raise ValueError('segment-ps must be positive and no longer than production-ps')
            if round(args.segment_ps * 1000 / (args.timestep_fs * args.save_every)) < 8:
                raise ValueError('segment-ps requires at least eight saved samples')
            if not np.isfinite(args.laser_nm) or args.laser_nm <= 0:
                raise ValueError('laser-nm must be positive')
            if not np.isfinite(args.bond_scale) or args.bond_scale <= 0:
                raise ValueError('bond-scale must be positive')
            atoms = prepare(cfg)
            from .geometry import molecular_geometry
            preview = dict(natoms=len(atoms), volume_A3=atoms.get_volume(),
                nyquist_cm1=nyquist, bin_spacing_cm1=1e12 / (args.segment_ps * C_CM_S),
                saved_frames=round((args.production_ps + args.equilibration_ps)*1000/(args.timestep_fs*args.save_every))+1,
                spectroscopy='raman' if args.raman_only else 'raman-ir', config=cfg)
            if qcfg is not None:
                preview['qha'] = qcfg
                preview['volume_note'] = 'Input volume only; temperature-dependent volumes require the QHA calculation'
            if not args.raman_only:
                groups, whole, bonds = molecular_geometry(atoms, args.bond_scale)
                preview.update(molecules=int(groups.max())+1, molecule_sizes=np.bincount(groups).tolist(),
                               dipole_model='Neutral whole molecules; verify inferred molecule sizes and bonds')
            if args.dry_run:
                print(json.dumps(preview, indent=2))
                return
            if qcfg is not None:
                run_qha_series(args, cfg, qcfg)
                return
            if (args.output / 'series.json').exists():
                raise ValueError('This directory contains a QHA series; choose a separate fixed-cell run directory')
            args.output.mkdir(parents=True, exist_ok=True)
            # Each subprocess has its own model/runtime state. Failed MD cannot
            # fall through to spectra. Stage commands may also be used directly.
            command = [sys.executable, '-m', 'md2spectra', 'md', '--output', str(args.output)]
            command += command_options(cfg)
            subprocess.run(command, check=True)
            subprocess.run([args.response_python, '-m', 'md2spectra', 'label',
                '--trajectory', str(args.output / 'trajectory.traj'), '--output', str(args.output / 'response'),
                '--mdp-model', args.mdp_model, '--device', args.response_device or args.device,
                '--bond-scale', str(args.bond_scale)] + (['--raman-only'] if args.raman_only else []), check=True)
            from .spectra import analyze
            metadata = analyze(args.output / 'response' / 'labels.npz', args.output / 'spectra',
                               args.temperature, args.equilibration_ps, args.segment_ps, args.laser_nm, args.raman_only)
            save_json(args.output / 'complete.json', dict(status='completed', preparation=preview,
                                                         spectrum=metadata))
            print(f'Completed: {args.output.resolve()}')
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f'Error: {error}\n')
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, 'Workflow stopped in a subprocess; completed checkpoints are retained.\n')


if __name__ == '__main__':
    main()
