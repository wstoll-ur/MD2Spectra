"""Generate a synthetic periodic water box for interface checks, not validation."""
from pathlib import Path
from ase.build import molecule
from ase.io import write
atoms = molecule('H2O')
atoms.set_cell([7, 7, 7])
atoms.pbc = True
atoms.center()
write(Path(__file__).with_name('synthetic-water.cif'), atoms)
