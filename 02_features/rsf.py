import numpy as np
from ovito.data import NearestNeighborFinder, CutoffNeighborFinder


def calculate_all_rsf(N_b_list, r_mults, sigma_mult, data):
    """
    Calculates the RSF based feature vector portion. The feature
    vector portion consists of a collection of G^{N_b}_{r,sigma} over
    all N_b in N_b_list and r,sigma which is determined by r_mults and
    the average distance to N_b nearest neighbors.

    Args:
        N_b_list (any iterable of ints): Values of N_b.
        r_mults (any iterable of ints): Values that we scale average
        r by for each atom to use for the r and sigma values of G^{N_b}_{r,sigma}.
        data (OVITO data object): Information about all atoms.

    Returns:
        A numpy 2d array storing the RSF part of the feature vectors.
    """
    # 1) Initialize
    num_atoms = data.particles.count
    feature_vec = [[] for _ in range(num_atoms)]

    # 2) Calculate
    for N_b in N_b_list:
        # A) Find average distances which we will use to normalize
        finder = NearestNeighborFinder(N_b, data)
        r_avg = []

        for atom in range(num_atoms):
            tot_dist = 0
            num_consider = 0

            for neigh in finder.find(atom):
                tot_dist += neigh.distance
                num_consider += 1

            r_avg.append(tot_dist / num_consider)

        # B) Calculate r_cut
        r_cut = max(r_avg) + 4 * max(r_avg) * sigma_mult

        # C) Calculate features for each atom
        finder = CutoffNeighborFinder(r_cut, data)

        for atom in range(num_atoms):
            dists = np.array([neigh.distance for neigh in finder.find(atom)])

            for r_mult in np.array(r_mults):
                r_use = r_avg[atom] * r_mult
                sigma = sigma_mult * r_avg[atom]

                g_val = np.sum(np.exp(-np.square(dists - r_use) / (2 * sigma**2)))
                feature_vec[atom].append(g_val)

    return np.array(feature_vec)
