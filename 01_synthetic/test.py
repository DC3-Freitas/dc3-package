from gen import LatticeGenerator
from cubic import cubic_lattice
import numpy as np
import vis

# Generate a perfect cubic lattice
lattice = cubic_lattice(1, 5)
#vis.plot_lattice(lattice)

# Perturb
generator = LatticeGenerator()
generator.load_np(lattice)
displaced_lattice = generator.generate(0.05).particles['Position']

# Visualize
vis.plot_lattice(displaced_lattice)