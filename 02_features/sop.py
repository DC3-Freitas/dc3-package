import numpy as np
from scipy.special import sph_harm_y
from ovito.data import NearestNeighborFinder


def sop_formula(l, thetas, phis):
    """
    Computes Q^{N_b}_{l} for a given atom whose neighbors are
    described by thetas and phis.

    Args:
        l (int): Value describing spherical harmonics.
        thetas (numpy array of floats): Polar angles.
        phis (numpy array of floats): Azimuthal angles.
    Returns:
        The value of Q^{N_b}_{l} described by a float.
    """
    # 1) Iterate over each m and sum up |q^{N_b}_{l,m}|^2
    N_b = len(thetas)
    q_accum = 0

    for m in range(-l, l + 1):
        # In the newer version (i.e. sph_harm_y), theta is polar and phi is azimuthal
        q = np.sum(sph_harm_y(m, l, thetas, phis)) / N_b
        q_accum += np.linalg.norm(q) ** 2

    # 2) Now that we have main summation, apply rest of terms to get Q^{N_b}_l
    return np.sqrt((4 * np.pi) / (2 * l + 1) * q_accum)


def sop_single_atom(N_b_list, l_list, neighbors):
    """
    Computes the feature vector for a single atom.

    The feature vector consists of a collection of
    Q^{N_b}_l over all N_b in N_b_list and l's in l_list.

    Args:
        N_b_list (any iterable of ints): Values of N_b.
        l_list (any iterable of ints): Values of l.
        neighbors (iterable of OVITO NearestNeighborFinder): Neighbors of given atom.

    Returns:
        A numpy array of floats storing a feature vector of the
        various values of Q^{N_b}_l over N_b in N_b_list and l in l_list.
    """
    # 1) Extract unit vectors and convert to spherical (neighbors guarenteed to be in sorted order)
    unit_vecs = np.array(
        [neigh.delta / np.linalg.norm(neigh.delta) for neigh in neighbors]
    )
    x_coords, y_coords, z_coords = unit_vecs[:, 0], unit_vecs[:, 1], unit_vecs[:, 2]
    thetas = np.arccos(z_coords)
    phis = np.arctan2(y_coords, x_coords)

    # 2) Calculate each Q^N_b_{l}
    Q = []

    for N_b in N_b_list:
        for l in l_list:
            Q.append(sop_formula(l, thetas[:N_b], phis[:N_b]))

    return np.array(Q)


def calculate_all_sop(N_b_list, l_list, data):
    """
    Calculates the Steindhart Order Parameters part of the
    feature vector for each atom. The feature vector portion consists
    of a collection of Q^{N_b}_l over all N_b in N_b_list and l's in l_list.

    Args:
        N_b_list (any iterable of ints): Values of N_b.
        l_list (any iterable of ints): Values of l.
        data (OVITO data object): Information about all atoms.

    Returns:
        A numpy 2d array storing the Steindhart Order Parameters part of the
        feature vectors.
    """
    # 1) Initialize variables
    num_atoms = data.particles.count
    feature_vec = []
    finder = NearestNeighborFinder(max(N_b_list), data)

    # 2) Iterate over each atom and compute sop vector for it
    for atom in range(num_atoms):
        feature_vec.append(sop_single_atom(N_b_list, l_list, finder.find(atom)))

    return np.array(feature_vec)
