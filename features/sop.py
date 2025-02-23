import numpy as np
from scipy.special import sph_harm_y
from ovito.data import NearestNeighborFinder
from tqdm import tqdm


def sop_single_atom(N_b_list, l_list, neighbors):
    """
    Computes the feature vector for a single atom.

    The feature vector consists of a collection of
    Q^{N_b}_l over all N_b in N_b_list and l's in l_list.

    Args:
        N_b_list (any SORTED iterable of ints): Values of N_b.
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

    for l in l_list:
        curAtom = 0

        # A tricky math thing is that we need to keep track of all different little q at once. Since we have
        # to divide each little q by N_b (accumulating all and then dividing by N_b is not the same)
        q_accum_all = np.zeros(2 * l + 1, dtype=np.complex128)

        for N_b in N_b_list:
            while (curAtom < N_b):
                q_accum_all += sph_harm_y(np.arange(-l, l + 1), l, thetas[curAtom], phis[curAtom])
                curAtom += 1

            Q.append(np.sqrt((4 * np.pi) / (2 * l + 1) * (np.linalg.norm(q_accum_all / N_b) ** 2)))
    
    # 3) Rearrange elements so that its flattened version of shape=(N_b, l)
    return np.array(Q).reshape((len(l_list), len(N_b_list))).T.flatten()


def calculate_all_sop(N_b_list, l_list, data):
    """
    Calculates the Steindhart Order Parameters part of the
    feature vector for each atom. The feature vector portion consists
    of a collection of Q^{N_b}_l over all N_b in N_b_list and l's in l_list.

    Args:
        N_b_list (any SORTED iterable of ints): Values of N_b.
        l_list (any iterable of ints): Values of l.
        data (OVITO data object): Information about all atoms.

    Returns:
        A numpy 2d array storing the Steindhart Order Parameters part of the
        feature vectors.
    """
    # 1) Initialize variables
    assert N_b_list == sorted(N_b_list), "N_b_list should be sorted"

    num_atoms = data.particles.count
    feature_vec = []
    finder = NearestNeighborFinder(max(N_b_list), data)

    # 2) Iterate over each atom and compute sop vector for it
    for atom in tqdm(range(num_atoms)):
        feature_vec.append(sop_single_atom(N_b_list, l_list, finder.find(atom)))

    return np.array(feature_vec)
