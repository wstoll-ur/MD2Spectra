import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from ase import units
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write
from ase.io.trajectory import Trajectory

from md2spectra.cli import main
from md2spectra.dynamics import run_md
from md2spectra.io import save_json, sha256
from md2spectra.qha import (FixedVolumeCellFilter, _relax_shape, _phonopy_from_cache, _phonons, fit_equilibrium,
                           gamma_optical_energies, relax_reference, run_qha, validate_qha, vibrational_free_energy)


class ThermodynamicsTests(unittest.TestCase):
    def test_gamma_only_removes_exactly_translations_and_rejects_optical_instability(self):
        gamma = [-1e-9, 0., 1e-9, .005, .01, .02]
        np.testing.assert_array_equal(gamma_optical_energies(gamma), [[.005, .01, .02]])
        for bad in [[-.01, 0, 0, .01], [-1e-9, 0, 0, 0], [0, 0, 0, np.nan]]:
            with self.assertRaises(ValueError):
                gamma_optical_energies(bad)

    def test_even_supercell_boundary_bond_matches_analytic_dispersion(self):
        # Strong dimers cross the unit-cell boundary; weak springs join equal
        # basis atoms in all directions. This is positive semidefinite exactly,
        # including at q points not commensurate with the 2x2x2 force supercell.
        from phonopy.physical_units import get_physical_units
        atoms = Atoms('C2', scaled_positions=[[.1, .3, .4], [.9, .3, .4]],
                      cell=[4, 5, 6], pbc=True)
        strong, weak, displacement = 10., .1, .01
        for side in [2, 3, 4]:
            shape = (side,)*3
            cache = {}
            def index(cell, basis):
                return np.ravel_multi_index(np.asarray(cell) % shape, shape) * 2 + basis
            for basis in range(2):
                for axis, letter in enumerate('xyz'):
                    for sign, suffix in [(-1, '-'), (1, '+')]:
                        force = np.zeros((2 * np.prod(shape), 3))
                        u = sign * displacement
                        force[basis, axis] = -(strong + 6 * weak) * u
                        force[index([-1 if basis == 0 else 1, 0, 0], 1-basis), axis] += strong*u
                        for direction in np.eye(3, dtype=int):
                            for step in [-1, 1]:
                                force[index(step*direction, basis), axis] += weak*u
                        cache[f'{basis}{letter}{suffix}'] = dict(forces=force)
            model = _phonopy_from_cache(SimpleNamespace(atoms=atoms, supercell=shape,
                                                        delta=displacement, cache=cache))
            q = np.array([[.125, .125, .125], [.3, -.2, .1], [.5, 0, 0]])
            measured = model.run_qpoints(q).frequencies
            acoustic = 2*weak*np.sum(1-np.cos(2*np.pi*q), axis=1)
            eigenvalues = np.repeat(np.column_stack([acoustic, acoustic+2*strong]), 3, axis=1)
            expected = np.sqrt(eigenvalues/atoms.get_masses()[0]) * get_physical_units().DefaultToTHz
            np.testing.assert_allclose(measured, expected, rtol=1e-10, atol=1e-10)

    def test_quantum_free_energy_and_q_weights(self):
        energy, temperature = .02, 300.
        modes = np.full((8, 6), energy)
        expected = 6 * (energy/2 + units.kB*temperature*np.log(1-np.exp(-energy/(units.kB*temperature))))
        free = vibrational_free_energy(modes, [0., temperature])
        np.testing.assert_allclose(free, [3*energy, expected])
        np.testing.assert_allclose(vibrational_free_energy(np.tile(modes, (2, 1)), [0., temperature]), free)
        # Very low frequencies and temperatures must not cause cancellation/overflow.
        self.assertTrue(np.isfinite(vibrational_free_energy([[1e-12, .2]], [1e-6, 1000])).all())

    def test_unstable_modes_are_not_silently_removed(self):
        for modes in [[[.02, -.001]], [[.02, 0]], [[np.nan]], [[np.inf]]]:
            with self.assertRaisesRegex(ValueError, 'phonon'):
                vibrational_free_energy(modes, [300])

    def test_minimum_pressure_and_boundary(self):
        volumes = np.linspace(90, 110, 9)
        energy = .01*(volumes-101)**2 - 1e4
        fitted = fit_equilibrium(volumes, energy)
        self.assertAlmostEqual(fitted['volume_A3'], 101, places=8)
        self.assertAlmostEqual(fitted['bulk_modulus_GPa'], 101*.02/units.GPa, places=6)
        pressured = fit_equilibrium(volumes, energy + units.GPa*volumes)
        self.assertAlmostEqual(pressured['volume_A3'], 101-units.GPa/.02, places=8)
        for bad in [.01*(volumes-120)**2, -.01*(volumes-100)**2]:
            with self.assertRaises(ValueError):
                fit_equilibrium(volumes, bad)

    def test_volume_softening_gives_thermal_expansion(self):
        volumes = np.linspace(90, 110, 9)
        free = np.array([vibrational_free_energy(np.full((8, 6), .02*100/v), [0, 300, 600])
                         for v in volumes]) + (.01*(volumes-100)**2)[:, None]
        equilibrium = [fit_equilibrium(volumes, free[:, i])['volume_A3'] for i in range(3)]
        self.assertTrue(np.all(np.diff(equilibrium) > 0))


class QHAWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cif = self.root / 'copper.cif'
        write(self.cif, bulk('Cu', 'fcc', a=3.6, cubic=True))
        self.cfg = dict(cif=str(self.cif), cif_index=0, uma_model='test', task='omat', device='cpu',
                        temperatures=[0., 300.], volume_factors=[.94, .98, 1.02, 1.06, 1.10],
                        phonon_supercell=[2, 2, 2], mesh=[2, 2, 2], displacement=.01,
                        fmax=.001, pressure_gpa=0.)

    def tearDown(self):
        self.tmp.cleanup()

    def test_validation(self):
        for key, value in [('temperatures', [300, 200]), ('temperatures', [0, 0]),
                           ('mesh', [3, 2, 2]), ('volume_factors', [.9, 1., 1.1]),
                           ('volume_factors', [.9, .95, 1., 1., 1.1]),
                           ('phonon_supercell', [0, 1, 1]), ('pressure_gpa', np.nan),
                           ('displacement', 0), ('fmax', -1), ('shape_stress_gpa', 0)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_qha(dict(self.cfg, **{key: value}))

    def test_gamma_sampling_evaluates_only_gamma(self):
        atoms = read(self.cif)
        def qpoints(points):
            np.testing.assert_array_equal(points, np.zeros((1, 3)))
            frequencies = np.ones((1, 3*len(atoms)))
            frequencies[0, :3] = 0.
            return SimpleNamespace(frequencies=frequencies)
        model = SimpleNamespace(run_qpoints=qpoints)
        with patch('md2spectra.qha.Phonons'), patch('md2spectra.qha._phonopy_from_cache', return_value=model):
            energies = _phonons(atoms, None, self.root, dict(self.cfg, phonon_sampling='gamma'))
        self.assertEqual(energies.shape, (1, 3*len(atoms)-3))
        self.assertTrue((energies > 0).all())

    def test_gamma_qha_integration_and_cache_identity(self):
        cfg = dict(self.cfg, phonon_sampling='gamma')
        output = self.root/'gamma'
        result = run_qha(cfg, output, calculator=EMT())
        self.assertEqual(result['phonon_sampling'], 'gamma')
        self.assertIn('approximation', result['vibrational_model'])
        with np.load(output/'volume-000/phonons.npz') as data:
            self.assertEqual(data['energies_eV'].shape, (1, 9))
        with patch('md2spectra.qha._phonons', side_effect=AssertionError('reuse cache')):
            self.assertEqual(run_qha(cfg, output, calculator=EMT()), result)
        with self.assertRaisesRegex(ValueError, 'changed'):
            run_qha(dict(cfg, phonon_sampling='mesh'), output, calculator=EMT())

    def test_commensurate_instability_is_rejected_even_when_shifted_mesh_is_positive(self):
        atoms = read(self.cif)
        def qpoints(points):
            frequencies = np.ones((len(points), 3*len(atoms)))
            frequencies[0, :3] = 0.
            frequencies[1+int(np.prod(self.cfg['mesh'])), 0] = -.2
            return SimpleNamespace(frequencies=frequencies)
        model = SimpleNamespace(run_qpoints=qpoints)
        with patch('md2spectra.qha.Phonons'), patch('md2spectra.qha._phonopy_from_cache', return_value=model):
            with self.assertRaisesRegex(ValueError, 'supercell-commensurate'):
                _phonons(atoms, None, self.root, self.cfg)
        diagnostic = json.loads((self.root/'phonon-diagnostics.json').read_text())
        self.assertEqual(diagnostic['nonpositive_mesh_modes'], 0)
        self.assertEqual(diagnostic['nonpositive_commensurate_modes'], 1)

    def test_real_ase_phonons_restart_and_md_cell(self):
        output = self.root / 'qha'
        self.cfg['relax_reference'] = True
        result = run_qha(self.cfg, output, calculator=EMT())
        self.assertIn('reference/relaxed.extxyz', result['files'])
        self.assertAlmostEqual(result['reference_volume_A3'],
                               read(output/'reference/relaxed.extxyz').get_volume(), places=8)
        cold, hot = result['temperatures']
        self.assertGreater(hot['volume_A3'], cold['volume_A3'])
        atoms = read(output / hot['structure'])
        self.assertAlmostEqual(atoms.get_volume(), hot['volume_A3'], places=5)
        np.testing.assert_allclose(atoms.cell.angles(), [90, 90, 90])
        with patch('md2spectra.qha._phonons', side_effect=AssertionError('must reuse completed phonons')):
            self.assertEqual(run_qha(self.cfg, output, calculator=EMT()), result)
        with self.assertRaisesRegex(ValueError, 'changed'):
            run_qha(dict(self.cfg, temperatures=[100, 300]), output, calculator=EMT())
        cfg = dict(cif=str(output / hot['structure']), cif_index=0, temperature=300., supercell=[2, 1, 1],
                   production_ps=.008, equilibration_ps=0., timestep_fs=.25, save_every=4,
                   thermostat_fs=100., checkpoint_ps=.004, seed=123, uma_model='test',
                   task='omat', device='cpu', relax=False, fmax=.02)
        self.assertTrue(run_md(cfg, self.root / 'md', calculator=EMT()))
        with Trajectory(self.root / 'md' / 'trajectory.traj') as trajectory:
            for frame in trajectory:
                self.assertAlmostEqual(frame.get_volume(), 2*hot['volume_A3'], places=5)
                np.testing.assert_allclose(frame.cell, atoms.repeat([2, 1, 1]).cell, atol=1e-12, rtol=0)
        (output / hot['structure']).write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            run_qha(self.cfg, output, calculator=EMT())

    def test_partial_volume_restart(self):
        from md2spectra.qha import _phonons
        count = 0

        def interrupted(*args):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('interrupted')
            return _phonons(*args)

        output = self.root / 'partial'
        with patch('md2spectra.qha._phonons', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                run_qha(self.cfg, output, calculator=EMT())
        with patch('md2spectra.qha._phonons', wraps=_phonons) as resumed:
            run_qha(self.cfg, output, calculator=EMT())
            self.assertEqual(resumed.call_count, len(self.cfg['volume_factors']) - 1)


class ConstantVolumeShapeTests(unittest.TestCase):
    def test_full_reference_relaxes_volume_shape_positions_and_restarts(self):
        atoms = self.distorted_copper()
        atoms.set_cell(atoms.cell*1.04, scale_atoms=True)
        initial = atoms.copy()
        atoms.calc = EMT()
        initial_energy = atoms.get_potential_energy()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = relax_reference(atoms, EMT(), folder)
            record = json.loads((folder/'relaxed.json').read_text())
            self.assertEqual(len(result), len(initial))
            self.assertLess(result.get_volume(), initial.get_volume()*.95)
            self.assertLess(record['energy_eV'], initial_energy)
            self.assertLessEqual(record['max_force_eV_A'], .001)
            self.assertLessEqual(record['max_residual_stress_GPa'], .01)
            np.testing.assert_allclose(result.cell.angles(), 90., atol=.02)
            with patch.object(EMT, 'calculate', side_effect=AssertionError('reuse reference')):
                reused = relax_reference(initial, EMT(), folder)
            np.testing.assert_allclose(reused.cell, result.cell, atol=1e-12)
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                relax_reference(initial, EMT(), folder, stress_gpa=.02)
            (folder/'relaxed.extxyz').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                relax_reference(initial, EMT(), folder)

    def distorted_copper(self):
        atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
        deformation = np.array([[1.04, .05, 0], [0, .98, .035], [0, 0, 1/(1.04*.98)]])
        atoms.set_cell(atoms.cell @ deformation, scale_atoms=True)
        atoms.positions[0] += [.02, -.01, .01]
        return atoms

    def test_shape_and_positions_relax_at_exact_volume(self):
        class VolumeTrackingEMT(EMT):
            def calculate(self, atoms=None, *args, **kwargs):
                volumes.append(atoms.get_volume())
                return super().calculate(atoms, *args, **kwargs)

        volumes = []
        atoms = self.distorted_copper()
        initial_cell = atoms.cell.copy()
        target = atoms.get_volume()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = _relax_shape(atoms, VolumeTrackingEMT(), folder, target, .001, .01)
            np.testing.assert_allclose(volumes, target, rtol=1e-12, atol=0)
            self.assertGreater(np.max(abs(result.cell.lengths() - initial_cell.lengths())), .05)
            self.assertGreater(np.max(abs(result.cell.angles() - initial_cell.angles())), 1.)
            np.testing.assert_allclose(result.cell.angles(), 90., atol=.02)
            record = json.loads((folder/'relaxed.json').read_text())
            self.assertLess(record['energy_eV'], record['initial_energy_eV'])
            self.assertLessEqual(record['max_force_eV_A'], .001)
            self.assertLessEqual(record['max_deviatoric_stress_GPa'], .01)
            with patch.object(EMT, 'calculate', side_effect=AssertionError('should reuse shape')):
                reused = _relax_shape(atoms, EMT(), folder, target, .001, .01)
            np.testing.assert_allclose(reused.cell, result.cell, rtol=0, atol=1e-12)
            with self.assertRaisesRegex(ValueError, 'settings'):
                _relax_shape(atoms, EMT(), folder, target, .001, .02)
            cfg = dict(cif=str(folder/'relaxed.extxyz'), cif_index=0, temperature=300.,
                       supercell=[2, 1, 1], production_ps=.008, equilibration_ps=0.,
                       timestep_fs=.25, save_every=4, thermostat_fs=100., checkpoint_ps=.004,
                       seed=123, uma_model='test', task='omat', device='cpu', relax=False, fmax=.02)
            run_md(cfg, folder/'md', calculator=EMT())
            with Trajectory(folder/'md'/'trajectory.traj') as trajectory:
                for frame in trajectory:
                    np.testing.assert_allclose(frame.cell, reused.repeat([2, 1, 1]).cell, atol=1e-12, rtol=0)

    def test_filter_projects_volume_and_matches_energy_derivatives(self):
        atoms = self.distorted_copper()
        atoms.calc = EMT()
        target = atoms.get_volume()
        cell_filter = FixedVolumeCellFilter(atoms)
        coordinates = cell_filter.get_positions()
        # Include a hydrostatic trial step: the projection must remove it exactly.
        coordinates[-3:] += [[.08, .04, 0], [.04, -.01, .03], [0, .03, .05]]
        cell_filter.set_positions(coordinates)
        self.assertAlmostEqual(atoms.get_volume(), target, places=11)
        coordinates = cell_filter.get_positions()
        forces = cell_filter.get_forces()
        for row, column in [(0, 0), (len(atoms), 0), (len(atoms), 1), (len(atoms)+1, 2)]:
            delta = np.zeros_like(coordinates)
            delta[row, column] = 1e-6
            cell_filter.set_positions(coordinates+delta)
            plus = atoms.get_potential_energy()
            cell_filter.set_positions(coordinates-delta)
            minus = atoms.get_potential_energy()
            self.assertAlmostEqual(forces[row, column], -(plus-minus)/2e-6, places=5)
        cell_filter.set_positions(coordinates)

    def test_unconverged_shape_is_not_saved(self):
        atoms = self.distorted_copper()
        with tempfile.TemporaryDirectory() as tmp:
            with patch('md2spectra.qha.FIRE.irun', return_value=iter([False])):
                with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                    _relax_shape(atoms, EMT(), Path(tmp), atoms.get_volume(), .001, .01)
            self.assertFalse((Path(tmp)/'relaxed.json').exists())


class QHACliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cif = self.root / 'copper.cif'
        write(self.cif, bulk('Cu', 'fcc', a=3.6, cubic=True))
        self.base = ['run', '--cif', str(self.cif), '--output', str(self.root/'series'), '--raman-only']

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_range_and_existing_single_temperature(self):
        for options, expected in [(['--qha', '--temperature-range', '100', '300', '50'], [100, 150, 200, 250, 300]),
                                  (['--qha', '--temperatures', '100', '225', '300'], [100, 225, 300]),
                                  (['--temperature', '300'], None)]:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), patch('md2spectra.cli.subprocess.run') as execute:
                main(self.base + options + ['--dry-run'])
            execute.assert_not_called()
            preview = json.loads(stream.getvalue())
            if expected is not None:
                self.assertEqual(preview['qha']['temperatures'], expected)
            else:
                self.assertNotIn('qha', preview)
        self.assertFalse((self.root/'series').exists())

    def test_bad_ranges_fail_before_execution(self):
        for options in [['--qha', '--temperature-range', '100', '310', '50'],
                        ['--qha', '--temperature-range', '300', '100', '50'],
                        ['--qha', '--temperature-range', '100', '300', '0'],
                        ['--qha', '--temperature-range', '0', '300', '50'],
                        ['--temperature-range', '100', '300', '50'],
                        ['--qha', '--temperatures', '300', '100']]:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with patch('md2spectra.cli.subprocess.run') as execute, self.assertRaises(SystemExit) as error:
                    main(self.base + options)
                self.assertEqual(error.exception.code, 2)
                execute.assert_not_called()

    def test_standalone_allows_zero_kelvin(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            main(['qha', '--cif', str(self.cif), '--output', str(self.root/'qha'),
                  '--temperature-range', '0', '300', '100', '--dry-run'])
        self.assertEqual(json.loads(stream.getvalue())['config']['temperatures'], [0, 100, 200, 300])

    def fake_qha(self, command, check):
        if command[3] != 'qha':
            return
        output = Path(command[command.index('--output')+1])
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, temperature in enumerate([100., 200., 300.]):
            atoms = read(self.cif)
            atoms.set_cell(atoms.cell * (1+index*.01), scale_atoms=True)
            name = f'cell-{index}.extxyz'
            write(output/name, atoms)
            rows.append(dict(temperature_K=temperature, volume_A3=atoms.get_volume(),
                             structure=name, structure_sha256=sha256(output/name)))
        save_json(output/'qha-complete.json', dict(temperatures=rows))

    def test_series_passes_each_cell_and_temperature_to_existing_pipeline(self):
        with patch('md2spectra.cli.subprocess.run', side_effect=self.fake_qha) as execute:
            main(self.base + ['--qha', '--qha-relax-reference', '--qha-phonon-sampling', 'gamma', '--temperature-range', '100', '300', '100', '--supercell', '2', '1', '1'])
        self.assertEqual(execute.call_count, 4)
        self.assertIn('--qha-relax-reference', execute.call_args_list[0].args[0])
        qcommand = execute.call_args_list[0].args[0]
        self.assertEqual(qcommand[qcommand.index('--qha-phonon-sampling')+1], 'gamma')
        for index, call in enumerate(execute.call_args_list[1:]):
            command = call.args[0]
            self.assertEqual(command[3], 'run')
            self.assertEqual(float(command[command.index('--temperature')+1]), [100, 200, 300][index])
            self.assertTrue(command[command.index('--cif')+1].endswith(f'cell-{index}.extxyz'))
            self.assertEqual(command[command.index('--cif-index')+1], '0')
            self.assertIn('--raman-only', command)
            self.assertNotIn('--qha', command)
        done = json.loads((self.root/'series'/'complete.json').read_text())
        self.assertEqual(len(done['runs']), 3)

    def test_failed_qha_never_launches_md(self):
        with patch('md2spectra.cli.subprocess.run', side_effect=subprocess.CalledProcessError(2, 'qha')) as execute:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(self.base + ['--qha', '--temperature', '300'])
        self.assertEqual(execute.call_count, 1)
        self.assertFalse((self.root/'series'/'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
