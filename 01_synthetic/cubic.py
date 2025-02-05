# Generate a perfect cubic lattice

import numpy as np

# Create a simple cubic lattice
def cubic_lattice(a: float, n: int):
    """
    Generate a simple cubic lattice with lattice constant a and n atoms per side
    - a: lattice constant
    - n: number of atoms per side
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