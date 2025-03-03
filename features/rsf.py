import numpy as np
import numba as nb
from ovito.data import NearestNeighborFinder, CutoffNeighborFinder
from tqdm import tqdm


@nb.njit
def calc_group_g(dists, avg, r_mults, sigma_mult):
    """
    Calculates RSF over various r_mults for a single atom.

    Args:
        dists (numpy array): Distances to each neighboring atom within r_cut
        avg (float): Average distance to the N_b neighboring atoms
        r_mults (any iterable of ints): Values that we scale average r by for each atom to 
                                        use for the r and sigma values of G_{r,sigma}^{N_b}
        sigma_mult (float): Divide by this in exponent and used to determine r_cut
    Returns:
        A numpy array storing RSF for the r_mults
    """
    fvec = np.zeros(len(r_mults))

    for i, r_mult in enumerate(r_mults):
        r_use = avg * r_mult
        sigma = sigma_mult * avg

        g_val = np.sum(np.exp(-np.square(dists - r_use) / (2 * (sigma**2))))
        fvec[i] = g_val

    return fvec


@nb.njit
def calc_rsf_single_atom(r_avg, r_cuts, all_dists, r_mults, sigma_mult):
    """
    Calculates RSF values for a single atom.

    Args:
        r_avg (numpy array): Averages distances for each N_b in 
                             n_b_list (n_b_list defined in calculate_all_rsf)
        r_cuts (numpy array): Value of r_cut for each N_b
        all_dists (SORTED iterable of ints): Sorted list of distances (within max r_cut) 
                                             from the given atom
        r_mults (any iterable of ints): Values that we scale average r by for each 
                                        atom to use for the r and sigma values of G_{r,sigma}^{N_b}
        sigma_mult (float): Divide by this in exponent and used to determine r_cut

    Returns:
        A numpy array storing RSF for the given atom
    """
    fvec = np.zeros(len(r_avg) * len(r_mults))
    dists = []
    n_b_idx = 0

    # Process as we iterate over neighbors
    for dist in all_dists:
        while n_b_idx < len(r_avg) and dist > r_cuts[n_b_idx]:
            fvec[n_b_idx * len(r_mults) : (n_b_idx + 1) * len(r_mults)] = calc_group_g(
                np.array(dists), r_avg[n_b_idx], r_mults, sigma_mult
            )
            n_b_idx += 1
        dists.append(dist)

    # Process remaining
    while n_b_idx < len(r_avg):
        fvec[n_b_idx * len(r_mults) : (n_b_idx + 1) * len(r_mults)] = calc_group_g(
            np.array(dists), r_avg[n_b_idx], r_mults, sigma_mult
        )
        n_b_idx += 1

    return fvec


def calculate_all_rsf(n_b_list, r_mults, sigma_mult, data):
    """
    Calculates the RSF based feature vector portion. The feature
    vector portion consists of a collection of G^{N_b}_{r,sigma} over
    all N_b in n_b_list and r,sigma which is determined by r_mults and
    the average distance to N_b nearest neighbors.

    Make sure that the number of atoms is at least max(n_b_list).

    Args:
        n_b_list (any SORTED iterable of ints): Values of N_b.
        r_mults (any iterable of ints): Values that we scale average r by for each atom to use
                                        for the r and sigma values of G_{r,sigma}^{N_b}.
        sigma_mult (float): Divide by this in exponent and used to determine r_cut
        data (OVITO data object): Information about all atoms.

    Returns:
        A numpy 2d array storing the RSF part of the feature vectors.
    """
    # 1) Initialize
    assert n_b_list == sorted(n_b_list), "n_b_list should be sorted"

    num_atoms = data.particles.count
    feature_vec = [[] for _ in range(num_atoms)]

    # 2) Calculate average distances for each (atom, N_b)
    r_avgs = np.zeros((num_atoms, len(n_b_list)))
    finder_avg = NearestNeighborFinder(max(n_b_list), data)

    for atom in tqdm(range(num_atoms), "RSF: Preprocessing"):
        tot_dist = 0
        num_consider = 0
        n_b_idx = 0

        for neigh in finder_avg.find(atom):
            tot_dist += neigh.distance
            num_consider += 1

            while n_b_idx < len(n_b_list) and num_consider == n_b_list[n_b_idx]:
                r_avgs[atom][n_b_idx] = tot_dist / num_consider
                n_b_idx += 1

        # If there are too few atoms
        while n_b_idx < len(n_b_list) and num_consider == n_b_list[n_b_idx]:
            r_avgs[atom][n_b_idx] = tot_dist / num_consider
            n_b_idx += 1

    # Take column-wise maximum to calculate r_cut for each N_b
    max_avgs = np.max(r_avgs, axis=0)
    r_cuts = max_avgs + 4 * max_avgs * sigma_mult

    # Avoid any floating point issues
    r_cuts = np.round(r_cuts, decimals=10)
    assert np.array_equal(r_cuts, np.sort(r_cuts)), "r_cut should be sorted"

    # 3) Calculate features
    finder_cutoff = CutoffNeighborFinder(max(r_cuts), data)

    for atom in tqdm(range(num_atoms), "RSF: Calculating"):
        # The neighbors here may not be in sorted order so we manually sort them
        all_dists = sorted([neigh.distance for neigh in finder_cutoff.find(atom)])
        feature_vec[atom] = calc_rsf_single_atom(
            r_avgs[atom], r_cuts, all_dists, r_mults, sigma_mult
        )

    return np.array(feature_vec)
