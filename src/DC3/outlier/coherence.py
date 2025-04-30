import numpy as np
from ovito.data import NearestNeighborFinder, DataCollection
from tqdm import tqdm
from DC3.compute_features.spherical_harmonics import (
    precalculate_sop_norm_factors,
    calc_spherical_harmonics,
)
import numba as nb
from DC3.constants import N_B_COHERENCE, L_ARR_COHERENCE, ALPHA_CUTOFF


@nb.njit(fastmath=True, cache=True)
def coherence_single_atom(
    l_arr: np.ndarray, unit_vecs: np.ndarray, norm_factors: np.ndarray
) -> np.ndarray:
    """
    Calculates vector (Xi) used to calculate vector used for coherence
    for a single atom.

    Args:
        l_arr: values of l to be included in the vectors
        unit_vecs: positions of atoms in array of shape (n, 3)
        norm_factors: precalculated values used in spherical harmonics
    Returns:
        A vector of size sum(l_arr * 2 + 1) used for coherence.

        More specifically, this vector is described as the concatenation
        of the vectors in the following list.

        [q^{N_b}_4, q^{N_b}_6, q^{N_b}_8, q^{N_b}_12]
    """
    # 1) Extract unit vectors and convert to spherical
    x_coords, y_coords, z_coords = unit_vecs[:, 0], unit_vecs[:, 1], unit_vecs[:, 2]
    thetas = np.arccos(z_coords)
    phis = np.arctan2(y_coords, x_coords)

    # 2) Calculate vectors
    e_vec = np.zeros((np.sum(l_arr * 2 + 1)), dtype=np.complex128)
    index = 0

    for l in l_arr:
        all_sph_harmonics = calc_spherical_harmonics(l, thetas, phis, norm_factors)
        e_vec[index : index + 2 * l + 1] += np.sum(all_sph_harmonics, axis=0) / len(
            unit_vecs
        )
        index += 2 * l + 1

    # 3) Normalize
    e_vec /= np.linalg.norm(e_vec)
    return e_vec


def calculate_all_coherence_values(
    n_b: int, l_arr: np.ndarray, data: DataCollection
) -> np.ndarray:
    """
    Calculates coherence parameter for each atom. The smaller it is,
    the more amorphous its arrangement of neighbors is.

    Args:
        N_neigh: number of neighbors we need to consider in our calculations
        l_arr: values of l to be considered
        data: object containing information about all the atoms
    Returns:
        Coherence parameter of each atom in the range [0, 1] with
        smaller values indicating a more amorphous arrangement of neighbors.
    """
    # 1) Initialize variables
    num_atoms = data.particles.count
    assert num_atoms >= n_b, "Must have sufficient number of atoms"

    finder = NearestNeighborFinder(n_b, data)
    norm_factors = precalculate_sop_norm_factors(max(l_arr))

    E_vec = np.zeros((num_atoms, np.sum(l_arr * 2 + 1)), dtype=np.complex128)
    coh_fac = np.zeros(num_atoms)

    # 2) Calculate all vectors
    for atom in tqdm(range(num_atoms), desc="Coherence: Vectors"):
        unit_vecs = np.array(
            [
                np.array(neigh.delta) / np.linalg.norm(neigh.delta)
                for neigh in finder.find(atom)
            ]
        )
        E_vec[atom] = coherence_single_atom(l_arr, unit_vecs, norm_factors)

    # 3) Calculate coherence via dot products
    for atom in range(num_atoms):
        for neighbor in finder.find(atom):
            # Hermetian inner product
            coh_fac[atom] += np.dot(E_vec[atom].conjugate(), E_vec[neighbor.index]).real

        coh_fac[atom] /= n_b

    return coh_fac


def calculate_amorphous(data: DataCollection) -> np.ndarray:
    """
    Classifies each atom in given data as amorphous or not amorphous.

    Args:
        data: OVITO DataCollection containing information about atoms
    Returns:
        A boolean array indicating whether each atom is amorphous (1 means amorphous, 0 means not amorphous)
    """
    return (
        calculate_all_coherence_values(N_B_COHERENCE, L_ARR_COHERENCE, data)
        < ALPHA_CUTOFF
    )
