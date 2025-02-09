"""
Compute t-SNE on synthetic dataset.
"""

from gen import LatticeGenerator
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

PCA_COMPONENTS = 50


def tsne(samples):
    """
    Compute t-SNE on a dataset.

    Args:
        samples: dictionary of labels and numpy arrays (n x d) of data points
    """
    # Generate data and label arrays
    total_samples = sum([len(samples[lattice_type]) for lattice_type in samples])
    sample_len = 512  # len(samples[list(samples.keys())[0]][0].particles["Position"])
    data = np.zeros((total_samples, sample_len * 3))
    labels = np.zeros(total_samples)
    ctr = 0
    m = {lattice_type: i for i, lattice_type in enumerate(samples)}
    bm = {i: lattice_type for i, lattice_type in enumerate(samples)}
    for lattice_type in samples:
        particles = np.array(
            [
                lattice.particles["Position"][:sample_len].flatten()
                for lattice in samples[lattice_type]
            ]
        )  # TODO determine why different samples have different number of atoms
        data[ctr : ctr + len(particles)] = particles
        labels[ctr : ctr + len(particles)] = np.full(len(particles), m[lattice_type])
        ctr += len(particles)

    # Run PCA for dimensionality reduction
    pca = PCA(n_components=PCA_COMPONENTS)
    data = pca.fit_transform(data)

    # Run t-SNE
    tsne = TSNE(n_components=len(samples), random_state=0)
    data_2d = tsne.fit_transform(data)

    # Plot
    plt.style.use("bmh")
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
    scatter_handles, scatter_labels = scatter.legend_elements()
    plt.legend(handles=scatter_handles, labels=[bm[i] for i in range(len(samples))])
    plt.show()


if __name__ == "__main__":
    sc_gen = LatticeGenerator()
    bcc_gen = LatticeGenerator()
    fcc_gen = LatticeGenerator()

    sc_gen.load_lammps("lammps_lattices/data/sc.gz")
    bcc_gen.load_lammps("lammps_lattices/data/bcc.gz")
    fcc_gen.load_lammps("lammps_lattices/data/fcc.gz")

    print("perfect lattices loaded!")

    samples = {
        "sc": sc_gen.generate_range(0, 0.25, 1000),
        "bcc": bcc_gen.generate_range(0, 0.25, 1000),
        "fcc": fcc_gen.generate_range(0, 0.25, 1000),
    }

    print("lattices generated!")

    tsne(samples)
