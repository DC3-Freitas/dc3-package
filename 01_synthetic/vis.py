"""
Visualize a synthetic dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
#import ovito


def plot_lattice(lattice: np.ndarray):
    """
    Plot a lattice in 3D.
    - lattice: expects a numpy array (n x 3) of lattice positions
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(lattice[:, 0], lattice[:, 1], lattice[:, 2])
    plt.show()

def plot_lattices(lattices: list[np.ndarray]):
    """
    Plot multiple lattices in 3D.
    - lattices: expects a list of numpy arrays (n x 3) of lattice positions
    """
    fig = plt.figure()
    # style
    plt.style.use("bmh")
    # multiple subplots
    for i, data in enumerate(lattices):
        lattice, title = data
        ax = fig.add_subplot(1, len(lattices), i+1, projection="3d")
        ax.scatter(lattice[:, 0], lattice[:, 1], lattice[:, 2])
        ax.set_title(title)
    plt.show()

def plot_tsne():
    """
    Plot a t-SNE visualization of the synthetic dataset.
    """
    pass