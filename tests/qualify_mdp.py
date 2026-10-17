"""Optional real-model qualification. Not collected by the model-free CI suite.

python tests/qualify_mdp.py --model /path/to/MACE-MDP.model --structure crystal.cif
"""
import argparse
import json
import numpy as np
from ase.io import read
from md2spectra.geometry import molecular_geometry
from md2spectra.response import response_calculator, qualify, predict

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--structure', required=True)
p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
a = p.parse_args()
atoms = read(a.structure, index=0)
calc = response_calculator(a.model, a.device)
result = qualify(calc, atoms, 1.15)
g, w, _ = molecular_geometry(atoms)
alpha, mu, _ = predict(calc, atoms, g, w)
repeated = atoms.repeat((2, 1, 1))
g, w, _ = molecular_geometry(repeated)
alpha2, mu2, _ = predict(calc, repeated, g, w)
np.testing.assert_allclose(alpha2, 2 * alpha, rtol=1e-6, atol=1e-7)
np.testing.assert_allclose(mu2, 2 * mu, rtol=1e-6, atol=1e-7)
result['repeated_cell_alpha_error'] = float(abs(alpha2 - 2 * alpha).max())
result['repeated_cell_dipole_error'] = float(abs(mu2 - 2 * mu).max())
print(json.dumps(result, indent=2))
