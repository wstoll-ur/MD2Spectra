"""Small, explicit provenance and atomic-write helpers."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def save_npz(path, **values):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def versions():
    out = {}
    for name in ['md2spectra', 'ase', 'numpy', 'scipy', 'torch', 'fairchem-core', 'mace-torch']:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


@contextlib.contextmanager
def lock(path):
    # A kernel lock releases automatically after crashes; Linux and macOS supported.
    import fcntl
    with Path(path).open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is using {path}') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
