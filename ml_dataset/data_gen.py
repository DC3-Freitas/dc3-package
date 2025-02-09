"""
Strings together `lattice` and `features` to create synthetic data
"""

from lattice.gen import LatticeGenerator
from features.compute_all import compute_feature_vectors
import numpy as np
from tqdm import tqdm

def create(structure_name, fname, alpha_min, alpha_max, num_lattices):
    """
    Creates numpy dataset of synthetic RSF/SOP data.

    Args:
        structure_name (str): Name of the structure to generate (saves output to `data/structure_name`)
        fname (str): Path to LAMMPS data file of perfect lattice in order to initialize generator
        alpha_min (float): Minimum alpha value for synthetic data
        alpha_max (float): Maximum alpha value for synthetic data
        num_lattices (int): Number of lattices to generate, linearly interpolated between alpha_min and alpha_max
    
    Returns:
        Nothing
    """

    # Initialize generator
    generator = LatticeGenerator()
    generator.load_lammps(fname)

    # Generate synthetic data
    data = []
    for displaced_lattice in tqdm(generator.generate_range(alpha_min, alpha_max, num_lattices)):
        data.append(compute_feature_vectors(displaced_lattice, None))
    data = np.hstack(data)
    np.savetxt(f"data/{structure_name}.gz", data)

if __name__ == "__main__":
    # Create FCC dataset
    print("== generating FCC dataset... ==")
    create("fcc", "lattice/lammps_lattices/data/fcc.gz", 0, 0.25, 40) # values from the paper
    # Create BCC dataset
    create("bcc", "lattice/lammps_lattices/data/bcc.gz", 0, 0.25, 40)
    # Create SC dataset
    create("sc", "lattice/lammps_lattices/data/sc.gz", 0, 0.25, 40)