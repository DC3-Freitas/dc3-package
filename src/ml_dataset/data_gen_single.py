"""
Strings together `lattice` and `features` to create synthetic data
"""

from lattice.gen import LatticeGenerator
from features.compute_all import compute_feature_vectors
import numpy as np
from tqdm import tqdm
import sys
import os


def create(structure_name, fname, alpha):
    """
    Creates numpy dataset of synthetic RSF/SOP data.

    Args:
        structure_name (str): Name of the structure to generate (saves output to `data/structure_name`)
        fname (str): Path to LAMMPS data file of perfect lattice in order to initialize generator
        alpha: Thermal displacement

    Returns:
        Nothing
    """
    print(f"Running {structure_name} on alpha={alpha}")

    # Initialize generator
    generator = LatticeGenerator()
    generator.load_lammps(fname)

    data = compute_feature_vectors(generator.generate(alpha), None)
    os.makedirs(f"ml_dataset/data/{structure_name}", exist_ok=True)
    np.save(f"ml_dataset/data/{structure_name}/{alpha:.3f}.npy", data)


if __name__ == "__main__":
    create(sys.argv[1], sys.argv[2], np.linspace(0.01, 0.25, 10)[int(sys.argv[3])])
