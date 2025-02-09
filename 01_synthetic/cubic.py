# Test script to generate a perfect cubic lattice

import numpy as np

# Create a simple cubic lattice
def cubic_lattice(a: float, n: int):
    """
    Generate a simple cubic lattice with lattice constant a and n atoms per side
    
    Args:
        a: lattice constant
        n: number of atoms per side
    
    Returns:
        lattice: a numpy array of shape (n^3, 3) containing the positions of the atoms
    """
    lattice = np.array(
        [
            [i, j, k]
            for i in np.linspace(0, a * (n - 1), n)
            for j in np.linspace(0, a * (n - 1), n)
            for k in np.linspace(0, a * (n - 1), n)
        ]
    )
    return lattice