import numpy as np
import os
from DC3.compute_features.compute_all import compute_feature_vectors
from DC3.lattice.gen import LatticeGenerator
from DC3.constants import TEMPS


def create(
    lattice_path: str, alpha: float, structure: str | None, save_dir: list[str] | None
) -> tuple[str, np.ndarray]:
    """
    Creates single synthetic data based on perfect lattice.
    If path is provided, also saves a file containing the data.

    Args:
        lattice_path: path to the LAMMPS file containing the perfect lattice
        alpha: thermal displacement magnitude
        structure: optional structure label to associate with the generated data
        save_dir: if provided, the directory to save the generated features as a .npy file
    Returns:
        Computed feature matrix and also the structure passed in (does it this way to
        make it easier to parallelize)
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


def generate_from_perfect_lattices(
    lattice_paths: list[str],
    structures: list[str] | None,
    save_dirs: list[str] | None = None,
) -> list[np.ndarray]:
    """
    Generates synthetic features for a set of lattices across multiple alphas.

    Args:
        lattice_paths: list of paths to perfect LAMMPS lattice files
        structures: optional list of structure names, one per lattice
        save_dirs: optional list of directories to save results, one per lattice
    Returns:
        List of (structure, feature matrix) pairs for all generated lattices.
    """
    runs = []

    for i, lattice_path in enumerate(lattice_paths):
        for temp in TEMPS:
            runs.append(
                (
                    lattice_path,
                    temp,
                    structures[i] if structures is not None else None,
                    save_dirs[i] if save_dirs is not None else None,
                )
            )

    # Unfortunately needs to be single threaded because OVITO GUI does not support multithreading
    return [create(*args) for args in runs]
