import numpy as np
import os
from DC3.compute_features.compute_all import compute_feature_vectors
from DC3.lattice.gen import LatticeGenerator
from DC3.constants import TEMPS

def create(lattice_path, alpha, structure: None | str, save_dir: None | list[str]):
    """
    Creates numpy dataset of synthetic RSF/SOP data.

    Args:
        TODO: FIX THIS
        alpha: Thermal displacement

    Returns:
        Nothing
    """
    # Initialize generator
    generator = LatticeGenerator()
    generator.load_lammps(lattice_path)
    features = compute_feature_vectors(generator.generate(alpha))

    # Save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{alpha:.3f}.npy"), features)
    
    return (structure, features)


def generate_from_perfect_lattices(lattice_paths: list[str], structures: None | list[str], save_dirs: None | list[str] = None) -> list[np.ndarray]:
    """
    TODO
    """
    runs = []

    for i, lattice_path in enumerate(lattice_paths):
        for temp in TEMPS:
            runs.append((lattice_path, temp, structures[i] if structures is not None else None, save_dirs[i] if save_dirs is not None else None))

    # Unfortunately needs to be single threaded because OVITO GUI does not support multithreading
    return [create(*args) for args in runs]