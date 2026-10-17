import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from md2spectra.dynamics import run_md, prepare, validate_config
from md2spectra.geometry import molecular_geometry, molecular_dipole
from md2spectra.motion import export_motion, filter_motion
from md2spectra.response import label
from md2spectra.spectra import calculate, analyze, C_CM_S


class Harmonic(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, reference):
        super().__init__()
        self.reference = reference.copy()
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = atoms.positions - self.reference
        self.results = dict(energy=float(.5 * np.sum(displacement**2)), forces=-displacement)


class MolecularResponse(Calculator):
    implemented_properties = ['polarizability', 'dipole', 'charges']
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        groups, whole, _ = molecular_geometry(atoms)
        q = np.where(atoms.numbers == 8, -.8, .4)
        alpha = np.eye(3) * len(atoms)
        for group in np.unique(groups):
            v = whole[groups == group]
            v = v - v.mean(axis=0)
            alpha += v.T @ v
        self.results = dict(polarizability=alpha, dipole=q @ atoms.positions, charges=q)


def water():
    atoms = molecule('H2O')
    atoms.set_cell([[7, 0, 0], [1.5, 7, 0], [.5, 1, 7]])
    atoms.pbc = True
    atoms.translate([3, 3, 3])
    return atoms


class SpectralTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(4000) * 1e-15
        self.wave = np.sin(2 * np.pi * 10e12 * self.time)
        self.tensor = np.array([[2, .3, .1], [.3, 1, -.2], [.1, -.2, 3]])
        self.alpha = self.wave[:, None, None] * self.tensor
        self.mu = self.wave[:, None] * np.array([1., 2., 3.])
    def calc(self, alpha=None, mu=None, **kw):
        return calculate(self.alpha if alpha is None else alpha, self.mu if mu is None else mu,
                         1., kw.get('temperature', 300.), kw.get('volume', 100.), 1.)
    def test_frequency_rotation_and_polarization(self):
        base = self.calc()
        peak = base['wavenumber_cm1'][np.argmax(base['raman_classical'])]
        self.assertAlmostEqual(peak, 10e12/C_CM_S)
        q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
        rotated = self.calc(q @ self.alpha @ q.T, self.mu @ q.T)
        for key in ['raman_classical', 'ir_absorption']:
            np.testing.assert_allclose(base[key], rotated[key], atol=np.max(base[key])*1e-12)
        iso = self.calc(self.wave[:, None, None] * np.eye(3))
        np.testing.assert_allclose(iso['raman_crossed'], 0, atol=1e-30)
    def test_scaling_temperature_and_constant(self):
        base = self.calc()
        repeated = self.calc(2*self.alpha, 2*self.mu, volume=200.)
        for key in ['raman_classical', 'ir_absorption']:
            np.testing.assert_allclose(repeated[key], 2*base[key])
        hotter = self.calc(temperature=600.)
        np.testing.assert_allclose(hotter['ir_absorption'], .5*base['ir_absorption'])
        z = self.calc(np.ones_like(self.alpha), np.ones_like(self.mu))
        np.testing.assert_array_equal(z['raman_classical'], 0)
        np.testing.assert_array_equal(z['ir_absorption'], 0)
    def test_ir_frequency_squared(self):
        mu = np.column_stack([np.sin(2*np.pi*10e12*self.time),
                              np.sin(2*np.pi*20e12*self.time), np.zeros(4000)])
        spec = self.calc(mu=mu)
        self.assertAlmostEqual(spec['ir_y'].max()/spec['ir_x'].max(), 4., places=10)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): self.calc(temperature=0)
        with self.assertRaises(ValueError): calculate(self.alpha[:5],self.mu[:5],1,300,100,1)
        bad = self.alpha.copy();bad[0,0,1] += 1
        with self.assertRaises(ValueError): self.calc(bad)


class GeometryTests(unittest.TestCase):
    def test_wrapping_neutral_dipole(self):
        a = water().repeat((2, 1, 1))
        q = np.tile([-.8, .4, .4], 2)
        g, whole, bonds = molecular_geometry(a)
        mu = molecular_dipole(q @ a.positions, q, a.positions, whole, g)
        a.positions += [6.9, 5.1, -7.8];a.wrap()
        g2, whole2, bonds2 = molecular_geometry(a)
        other = molecular_dipole(q @ a.positions, q, a.positions, whole2, g2)
        np.testing.assert_allclose(mu, other, atol=1e-13)
        np.testing.assert_array_equal(g, g2)
        self.assertEqual(bonds,bonds2)
    def test_periodic_network_rejected(self):
        with self.assertRaisesRegex(ValueError, 'network'):
            molecular_geometry(Atoms('C', positions=[[0,0,0]], cell=[1.4,5,5], pbc=True))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cif = self.root / 'water.cif';write(self.cif, water())
        self.cfg = dict(cif=str(self.cif), cif_index=0, temperature=100., supercell=[1,1,1],
            production_ps=.016, equilibration_ps=0., timestep_fs=.25, save_every=4,
            thermostat_fs=100., checkpoint_ps=.004, seed=123, uma_model='synthetic-test',
            task='omc', device='cpu', relax=False, fmax=.02)
    def tearDown(self): self.tmp.cleanup()
    def test_restart_discards_tail_and_preserves_rng(self):
        reference = prepare(self.cfg).positions
        full, split = self.root/'full', self.root/'split'
        self.assertTrue(run_md(self.cfg,full,Harmonic(reference)))
        self.assertFalse(run_md(self.cfg,split,Harmonic(reference),stop_after_steps=24))
        with Trajectory(split/'trajectory.traj','a') as tr:
            a = prepare(self.cfg);a.info=dict(step=28,time_ps=.007);tr.write(a)
        self.assertTrue(run_md(self.cfg,split,Harmonic(reference)))
        with Trajectory(full/'trajectory.traj') as a,Trajectory(split/'trajectory.traj') as b:
            self.assertEqual(len(a),len(b))
            for left,right in zip(a,b):
                np.testing.assert_allclose(left.positions,right.positions,atol=1e-14,rtol=0)
                np.testing.assert_allclose(left.get_momenta(),right.get_momenta(),atol=1e-14,rtol=0)
        changed=dict(self.cfg,temperature=200.)
        with self.assertRaisesRegex(ValueError,'changed'): run_md(changed,split,Harmonic(reference))
    def test_full_response_and_spectrum(self):
        folder=self.root/'run'
        run_md(self.cfg,folder,Harmonic(prepare(self.cfg).positions))
        labels=label(folder/'trajectory.traj',folder/'response',calculator=MolecularResponse(),chunk_size=8)
        meta=analyze(labels,folder/'spectra',100.,segment_ps=.008)
        self.assertEqual(meta['frames'],17)
        self.assertTrue((folder/'spectra'/'spectra.png').is_file())
        # Exercise checkpoint reuse after an interrupted labeling stage.
        (folder/'response'/'labels-complete.json').unlink()
        label(folder/'trajectory.traj',folder/'response',calculator=MolecularResponse(),chunk_size=8)
        with np.load(labels) as z:
            np.testing.assert_allclose(np.diff(z['time_ps']),.001)
        bad=self.root/'bad.npz'
        with np.load(labels) as z:
            values=dict(z);values['dipole_convention']='raw'
        np.savez(bad,**values)
        with self.assertRaisesRegex(ValueError,'raw dipoles'):analyze(bad,self.root/'bad',100,segment_ps=.008)
    def test_bad_sampling(self):
        with self.assertRaises(ValueError):validate_config(dict(self.cfg,production_ps=.0161))
        with self.assertRaises(ValueError):validate_config(dict(self.cfg,supercell=[0,1,1]))
    def test_motion_export_and_rejection(self):
        path=self.root/'motion.traj'
        times=np.arange(2000)*.001
        base=water();positions=[]
        with Trajectory(path,'w') as tr:
            for t in times:
                a=base.copy()
                a.positions[1,0] += .015*np.sin(2*np.pi*10*t)+.01*np.sin(2*np.pi*30*t)
                a.info['time_ps']=float(t)
                positions.append(a.positions.copy());tr.write(a)
        mean,filtered,spacing,_=filter_motion(np.array(positions),times,base.cell,base.get_masses(),300,370)
        freq=np.fft.rfftfreq(len(times),.001e-12)/C_CM_S
        f=np.abs(np.fft.rfft(filtered[:,1,0]))
        self.assertAlmostEqual(freq[np.argmax(f)],10e12/C_CM_S)
        self.assertLess(f[np.argmin(abs(freq-30e12/C_CM_S))],f.max()*1e-12)
        out=self.root/'band.extxyz'
        result=export_motion(path,out,300,370,clip_ps=.2,stride=2)
        frames=read(out,index=':')
        self.assertEqual(len(frames),result['exported_frames'])
        self.assertIn('displacement_A',frames[0].arrays)
        self.assertTrue(frames[0].info['visualization_only'])
        with self.assertRaisesRegex(ValueError,'already exists'):export_motion(path,out,300,370)
        with self.assertRaisesRegex(ValueError,'aliases'):export_motion(path,self.root/'bad.extxyz',300,370,stride=100)


if __name__ == '__main__':unittest.main()
