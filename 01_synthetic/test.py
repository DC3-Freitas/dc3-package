# Cubic visualization test script

from gen import LatticeGenerator
from cubic import cubic_lattice
import numpy as np
import vis

# Generate a perfect cubic lattice
lattice = cubic_lattice(1, 5)

# Perturb
generator = LatticeGenerator()
generator.load_np(lattice)
displaced_lattices = []
for alpha in np.linspace(0, 0.25, 5):
    displaced_lattices.append([
        generator.generate(alpha).particles['Position'],
        f"alpha = {alpha:.2f}"
    ])

# Visualize
vis.plot_lattices(displaced_lattices)