import numpy as np
from scipy.special import sph_harm_y
from ovito.data import NearestNeighborFinder

def sop_formula(l, thetas, phis):
    """
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
    """
    # 1) Extract unit vectors and convert to spherical (neighbors guarenteed to be in sorted order)
    unit_vecs = [neigh.delta / np.linalg.norm(neigh.delta) for neigh in neighbors]
    x_coords, y_coords, z_coords = unit_vecs[:, 0], unit_vecs[:, 1], unit_vecs[:, 2]
    thetas = np.arccos(z_coords)
    phis = np.arctan2(y_coords, x_coords)

    # 2) Calculate each Q^N_b_{l}
    Q = []

    for N_b in N_b_list: 
        for l in l_list:
            Q.append(sop_formula(l, thetas[: N_b + 1], phis[: N_b + 1]))
    
    return np.array(Q)

def calculate_all_sop(N_b_list, l_list, data):
    """
    """
    # 1) Initialize variables
    num_atoms = data.particles.count
    feature_vec = []
    finder = NearestNeighborFinder(max(N_b_list), data)

    # 2) Iterate over each atom and compute sop vector for it
    for atom in range(num_atoms):
        feature_vec.append(sop_single_atom(N_b_list, l_list, finder.find(atom)))

    return np.array(feature_vec)