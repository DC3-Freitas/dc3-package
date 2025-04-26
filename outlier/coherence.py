import numpy as np
from ovito.data import NearestNeighborFinder
from tqdm import tqdm
from features.sop import precalculate_sop_norm_factors, calc_spherical_harmonics
import numba as nb


@nb.njit
def coherence_single_atom(l_list, unit_vecs, norm_factors):
    # 1) Extract unit vectors and convert to spherical
    x_coords, y_coords, z_coords = unit_vecs[:, 0], unit_vecs[:, 1], unit_vecs[:, 2]
    thetas = np.arccos(z_coords)
    phis = np.arctan2(y_coords, x_coords)

    # 2) Calculate vectors
    E_vec = np.zeros((np.sum(np.array(l_list) * 2 + 1)), dtype=np.complex128)
    index = 0

    for l in l_list:
        all_sph_harmonics = calc_spherical_harmonics(l, thetas, phis, norm_factors)
        E_vec[index : index + 2 * l + 1] += np.sum(all_sph_harmonics, axis=0) / len(unit_vecs)
        index += 2 * l + 1

    # 3) Normalize
    E_vec /= np.linalg.norm(E_vec)
    return E_vec


def coherence(N_neigh, l_list, data):
    # 1) Initialize variables 
    finder = NearestNeighborFinder(N_neigh,data)
    num_atoms = data.particles.count
    norm_factors = precalculate_sop_norm_factors(max(l_list))

    E_vec = np.zeros((num_atoms, np.sum(np.array(l_list) * 2 + 1)), dtype=np.complex128)
    coh_fac = np.zeros(num_atoms)

    # 2) Calculate all vectors
    for atom in tqdm(range(num_atoms)):
        unit_vecs = np.array(
            [neigh.delta / np.linalg.norm(neigh.delta) for neigh in finder.find(atom)]
        )
        E_vec[atom] = coherence_single_atom(l_list, unit_vecs, norm_factors)

    # 3) Calculate coherence via dot products
    for atom in range(num_atoms):
        for neighbor in finder.find(atom):
            # Hermetian inner product
            coh_fac[atom] += np.dot(E_vec[atom].conjugate(), E_vec[neighbor.index]).real

        coh_fac[atom] /= N_neigh

    return coh_fac