"""Molecular identity, whole molecules, and fixed-cell trajectory validation."""
import numpy as np
from ase.geometry import find_mic
from ase.neighborlist import natural_cutoffs, neighbor_list


def molecular_geometry(atoms, bond_scale=1.15):
    """Return connected-component IDs, whole coordinates and a bond signature.

    Nonzero winding cycles identify periodic networks, for which the neutral
    molecular dipole approximation is undefined. Covalent-radius inference is
    deliberately exposed through bond_scale and saved for inspection.
    """
    i, j, shifts = neighbor_list('ijS', atoms, natural_cutoffs(atoms, mult=bond_scale))
    adjacency = [[] for _ in atoms]
    for a, b, s in zip(i, j, shifts):
        adjacency[a].append((int(b), s))
    labels = np.full(len(atoms), -1, dtype=int)
    offsets = np.zeros((len(atoms), 3), dtype=int)
    molecule = 0
    for root in range(len(atoms)):
        if labels[root] >= 0:
            continue
        labels[root] = molecule
        stack = [root]
        while stack:
            a = stack.pop()
            for b, shift in adjacency[a]:
                proposed = offsets[a] + shift
                if labels[b] < 0:
                    labels[b] = molecule
                    offsets[b] = proposed
                    stack.append(b)
                elif not np.array_equal(offsets[b], proposed):
                    raise ValueError('Periodic bonded network detected; neutral-molecule IR is unsupported')
        molecule += 1
    bonds = sorted({(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(i, j)})
    return labels, atoms.positions + offsets @ atoms.cell.array, bonds


def molecular_dipole(raw_dipole, charges, raw_positions, whole_positions, groups):
    """Remove global q*r and rebuild neutral-molecule q*(r-centre).

    A uniform charge offset cancels within each centred molecule, so the
    calculator's globally normalized charges and raw charges give the same result.
    This approximation excludes intermolecular charge transfer and ionic transport.
    """
    relative = np.array(whole_positions, dtype=float, copy=True)
    for group in np.unique(groups):
        selected = groups == group
        relative[selected] -= relative[selected].mean(axis=0)
    return np.asarray(raw_dipole) + np.einsum('i,ij->j', charges, relative - raw_positions)


def validate_frame(atoms, reference):
    if not np.array_equal(atoms.numbers, reference.numbers):
        raise ValueError('Atomic identity/order changed')
    if not atoms.pbc.all() or not np.allclose(atoms.cell, reference.cell, rtol=0, atol=1e-8):
        raise ValueError('Only fully periodic, fixed-cell trajectories are supported')
    if not np.isfinite(atoms.positions).all():
        raise ValueError('Nonfinite positions')


def unwrap_positions(positions, cell):
    # ASE minimum-image vectors handle skewed cells (fractional rounding does not).
    delta = np.diff(positions, axis=0)
    mic, _ = find_mic(delta.reshape(-1, 3), cell, pbc=True)
    return np.concatenate([positions[:1], positions[:1] + np.cumsum(mic.reshape(delta.shape), axis=0)])
