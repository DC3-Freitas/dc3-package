import numpy as np
import numba as nb
from ovito.data import NearestNeighborFinder, DataCollection
from tqdm import tqdm
from DC3.compute_features.spherical_harmonics import (
    precalculate_sop_norm_factors,
    calc_spherical_harmonics,
)


@nb.njit
def sop_single_atom(
    n_b_list: list[int],
    l_list: list[int],
    unit_vecs: np.ndarray,
    norm_factors: np.ndarray,
) -> np.ndarray:
    """
    Computes the feature vector for a single atom.

    The feature vector consists of a collection of
    Q^{N_b}_l over all N_b in n_b_list and l's in l_list.

    Args:
        n_b_list: values of N_b, must be sorted
        l_list: values of l
        unit_vecs: positions of atoms
        norm_factors: precalculated values used in spherical harmonics

    Returns:
        Feature vector of the various values of Q^{N_b}_l over
        N_b in n_b_list and l in l_list.
    """
    # 1) Extract unit vectors and convert to spherical (neighbors guarenteed to be in sorted order)
    x_coords, y_coords, z_coords = unit_vecs[:, 0], unit_vecs[:, 1], unit_vecs[:, 2]
    thetas = np.arccos(z_coords)
    phis = np.arctan2(y_coords, x_coords)

    # 2) Calculate each Q_l^{N_b}
    q_vals_all = []

    for l in l_list:
        curAtom = 0

        # A tricky math thing is that we need to keep track of all different little q at once
        # This is since norm(sum of vectors) is not the same as sum(norm of vectors)
        q_accum_all = np.zeros(2 * l + 1, dtype=np.complex128)
        all_sph_harmonics = calc_spherical_harmonics(l, thetas, phis, norm_factors)

        for n_b in n_b_list:
            while curAtom < n_b:
                q_accum_all += all_sph_harmonics[curAtom]
                curAtom += 1

            q_vals_all.append(
                np.sqrt(
                    (4 * np.pi) / (2 * l + 1) * (np.linalg.norm(q_accum_all / n_b) ** 2)
                )
            )

    # 3) Rearrange elements so that its flattened version of shape=(N_b, l)
    return np.array(q_vals_all).reshape((len(l_list), len(n_b_list))).T.flatten()


def calculate_all_sop(
    n_b_list: list[int], l_list: list[int], data: DataCollection
) -> np.ndarray:
    """
    Calculates the Steindhart Order Parameters part of the
    feature vector for each atom. The feature vector portion consists
    of a collection of Q_l^{N_b} over all N_b in n_b_list and l's in l_list.

    Args:
        n_b_list: values of N_b, must be sorted
        l_list: values of l.
        data: information about all atoms

    Returns:
        Steindhart Order Parameters portion of the feature vectors.
    """
    # 1) Initialize variables
    assert n_b_list == sorted(n_b_list), "n_b_list should be sorted"

    num_atoms = data.particles.count
    feature_vec = []
    finder = NearestNeighborFinder(max(n_b_list), data)

    norm_factors = precalculate_sop_norm_factors(max(l_list))

    # 2) Iterate over each atom and compute sop vector for it
    for atom in tqdm(range(num_atoms), desc="SOP: Calculating"):
        unit_vecs = np.array(
            [
                np.array(neigh.delta) / np.linalg.norm(np.array(neigh.delta))
                for neigh in finder.find(atom)
            ]
        )
        feature_vec.append(sop_single_atom(n_b_list, l_list, unit_vecs, norm_factors))

    return np.array(feature_vec)
