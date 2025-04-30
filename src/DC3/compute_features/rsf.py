"""
Generates Radial Structure Function (RSF) values for feature computation.
"""

import numpy as np
import numba as nb
from ovito.data import NearestNeighborFinder, CutoffNeighborFinder, DataCollection
from tqdm import tqdm


@nb.njit(fastmath=True, inline="always", cache=True)
def calc_group_g(
    dists: np.ndarray, avg: float, r_mults: np.ndarray, sigma_mult: float
) -> np.ndarray:
    """
    Calculates RSF over various r_mults for a single atom.

    Args:
        dists: distances to each neighboring atom within r_cut
        avg: average distance to the N_b neighboring atoms
        r_mults: values that we scale average r by for each atom to
                 use for the r and sigma values of G_{r,sigma}^{N_b}
        sigma_mult: divide by this in exponent and used to determine r_cut
    Returns:
        A numpy array storing RSF for the r_mults.
    """
    fvec = np.zeros(len(r_mults))

    for i, r_mult in enumerate(r_mults):
        r_use = avg * r_mult
        sigma = sigma_mult * avg

        g_val = np.sum(np.exp(-np.square(dists - r_use) / (2 * (sigma**2))))
        fvec[i] = g_val

    return fvec


@nb.njit(fastmath=True, cache=True)
def calc_rsf_single_atom(
    r_avg: np.ndarray,
    r_cuts: np.ndarray,
    all_dists: np.ndarray,
    r_mults: np.ndarray,
    sigma_mult: float,
) -> np.ndarray:
    """
    Calculates RSF values for a single atom.

    Args:
        r_avg: averages distances for each N_b
        r_cuts: value of r_cut for each N_b
        all_dists: sorted array of distances (within max r_cut) from the given atom
        r_mults: values that we scale average r by for each
                 atom to use for the r and sigma values of G_{r,sigma}^{N_b}
        sigma_mult: divide by this in exponent and used to determine r_cut

    Returns:
        A numpy array storing RSF for the given atom.
    """
    fvec = np.zeros(len(r_avg) * len(r_mults))
    dists = np.zeros((len(all_dists),))
    n_b_idx = 0

    # Process as we iterate over neighbors
    for i, dist in enumerate(all_dists):
        while n_b_idx < len(r_avg) and dist > r_cuts[n_b_idx]:
            fvec[n_b_idx * len(r_mults) : (n_b_idx + 1) * len(r_mults)] = calc_group_g(
                dists[:i], r_avg[n_b_idx], r_mults, sigma_mult
            )
            n_b_idx += 1
        dists[i] = dist

    # Process remaining
    while n_b_idx < len(r_avg):
        fvec[n_b_idx * len(r_mults) : (n_b_idx + 1) * len(r_mults)] = calc_group_g(
            dists, r_avg[n_b_idx], r_mults, sigma_mult
        )
        n_b_idx += 1

    return fvec


def calculate_all_rsf(
    n_b_arr: np.ndarray, r_mults: np.ndarray, sigma_mult: float, data: DataCollection
) -> np.ndarray:
    """
    Calculates the RSF based feature vector portion. The feature
    vector portion consists of a collection of G^{N_b}_{r,sigma} over
    all N_b in n_b_arr and r,sigma which is determined by r_mults and
    the average distance to N_b nearest neighbors.

    Make sure that the number of atoms is at least max(n_b_arr).

    Args:
        n_b_arr: values of N_b, must be sorted
        r_mults: values that we scale average r by for each atom to use
                 for the r and sigma values of G_{r,sigma}^{N_b}.
        sigma_mult: divide by this in exponent and used to determine r_cut
        data: information about all atoms.

    Returns:
        A numpy 2d array storing the RSF part of the feature vectors.
    """
    # 1) Initialize
    assert np.all(n_b_arr == np.sort(n_b_arr)), "n_b_arr should be sorted"

    num_atoms = data.particles.count
    assert num_atoms >= n_b_arr.max(), "Must have sufficient number of atoms"

    # 2) Calculate average distances for each (atom, N_b)
    r_avgs = np.zeros((num_atoms, len(n_b_arr)))
    finder_avg = NearestNeighborFinder(max(n_b_arr), data)

    for atom in tqdm(range(num_atoms), "RSF: Preprocessing"):
        tot_dist = 0.0
        num_consider = 0
        n_b_idx = 0

        for neigh in finder_avg.find(atom):
            tot_dist += neigh.distance
            num_consider += 1

            while n_b_idx < len(n_b_arr) and num_consider == n_b_arr[n_b_idx]:
                r_avgs[atom, n_b_idx] = tot_dist / num_consider
                n_b_idx += 1

        # If there are too few atoms
        while n_b_idx < len(n_b_arr) and num_consider == n_b_arr[n_b_idx]:
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
    feature_vec = np.zeros((num_atoms, len(n_b_arr) * len(r_mults)))

    for atom in tqdm(range(num_atoms), "RSF: Calculating"):
        # The neighbors here may not be in sorted order so we manually sort them
        all_dists = np.array(
            sorted([neigh.distance for neigh in finder_cutoff.find(atom)])
        )
        feature_vec[atom, :] = calc_rsf_single_atom(
            r_avgs[atom], r_cuts, all_dists, r_mults, sigma_mult
        )

    return feature_vec
