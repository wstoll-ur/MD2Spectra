"""Two-stage model execution keeps UMA and MACE dtype/device state isolated."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .io import save_json


def add_md_options(p):
    p.add_argument('--cif', required=True)
    p.add_argument('--cif-index', type=int, default=0, help='Zero-based structure index in a multi-block CIF')
    p.add_argument('--temperature', type=float, required=True, help='Kelvin')
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
    p.add_argument('--uma-model', default='uma-s-1p2p1', help='FAIRChem model ID or trusted local checkpoint')
    p.add_argument('--uma-inference', choices=['conservative', 'md'], default=None,
                   help='Defaults to conservative; md merges experts and disables activation checkpointing; benchmark GPU memory first')
    p.add_argument('--task', default='omc', choices=['omc', 'omat', 'omol', 'oc20', 'oc22', 'oc25', 'odac'])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--relax', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fmax', type=float, default=.02, help='Fixed-cell relaxation force threshold, eV/Angstrom')
    p.add_argument('--output', type=Path, required=True)


MD_KEYS = ['cif', 'cif_index', 'temperature', 'supercell', 'production_ps', 'equilibration_ps',
           'timestep_fs', 'save_every', 'thermostat_fs', 'checkpoint_ps', 'seed', 'uma_model',
           'task', 'device', 'relax', 'fmax', 'save_momenta', 'uma_inference']


def main(argv=None):
    parser = argparse.ArgumentParser(description='CIF → UMA MD → molecular Raman and THz/IR spectra')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Run the complete restartable workflow')
    add_md_options(run)
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
        if args.command == 'label':
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
            if not args.raman_only:
                groups, whole, bonds = molecular_geometry(atoms, args.bond_scale)
                preview.update(molecules=int(groups.max())+1, molecule_sizes=np.bincount(groups).tolist(),
                               dipole_model='Neutral whole molecules; verify inferred molecule sizes and bonds')
            if args.dry_run:
                print(json.dumps(preview, indent=2))
                return
            args.output.mkdir(parents=True, exist_ok=True)
            # Each subprocess has its own model/runtime state. Failed MD cannot
            # fall through to spectra. Stage commands may also be used directly.
            command = [sys.executable, '-m', 'md2spectra', 'md', '--output', str(args.output)]
            for key, value in cfg.items():
                flag = '--' + key.replace('_', '-')
                if isinstance(value, bool):
                    command.append(flag if value else '--no-' + key.replace('_', '-'))
                elif isinstance(value, list):
                    command += [flag, *map(str, value)]
                else:
                    command += [flag, str(value)]
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
