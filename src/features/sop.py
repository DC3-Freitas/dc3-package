import numpy as np
import numba as nb
from ovito.data import NearestNeighborFinder
from tqdm import tqdm
import math


def precalculate_sop_norm_factors(max_l):
    """
    Calculates the following part of spherical harmonics:
    sqrt[(2l + 1) / 4pi * (l - m)! / (l + m)!]
    for all l and m up to the given max_l.

    Args:
        max_l (int): Maximum l value we will ever consider for
                     use of the returned table
    Returns:
        Table where the entry at (l, m) represents the above equation
        evaluated at the given l and m.
    """
    norm_factors = np.zeros((max_l + 1, max_l + 1))

    for l in range(0, max_l + 1):
        # As discussed below, we only need m >= 0
        for m in range(l + 1):
            # Accurately precalculate the terms here to avoid overflow
            log_term = 0.5 * (
                math.log(2 * l + 1)
                - math.log(4 * math.pi)
                + math.lgamma(l - m + 1)
                - math.lgamma(l + m + 1)
            )
            norm_factors[l, m] = math.exp(log_term)

    return norm_factors


@nb.njit
def double_fact(i):
    """
    Calculates i!! and returns 1 if i < 0
    Args:
        i (int): Number to calculate double factorial of
    Returns:
        i! if i >= 0, otherwise 1
    """
    result = 1
    for x in range(i, 0, -2):
        result *= x
    return result


@nb.njit
def calc_spherical_harmonics(l, thetas, phis, norm_factors):
    """
    Calculates spherical harmonics for each atom described
    by thetas and phis for m=-l...+l

    **Note: due to numba optimization, double_fact(2 * l - 1) can
    overflow if l is large. The current l=16 maximum is as large
    as it can go without overflowing!**

    Args:
        l (float): Subscript of Y
        thetas (numpy array of floats): Polar angles
        phis (numpy array of floats): Azimuthal angles
    Returns:
        2d numpy array sph_y where sph_y[i] stores the
        spherical harmonic values for atom i (i-th neighbor) for
        m=-l...+l in that order.
    """
    # 1) Variables (let first dimension be m -- we tranpose sph_y later)
    sph_y = np.zeros((l * 2 + 1, len(thetas)), dtype=np.complex128)
    plm = np.zeros((l + 1, len(thetas)))

    # 2) Calculate associated legendre polynomial P_l^m
    # We use the reccurence: P_l^m = something in terms of P^{m+1}_l and P^{m+2}_l
    # Note: m = -l...l but only need to calculate 0...l due to properties

    # 2a) Base cases
    x = np.cos(thetas)

    plm[l] = (
        (1 if (l) % 2 == 0 else -1) * double_fact(2 * l - 1) * np.power(1 - x**2, l / 2)
    )

    if l - 1 >= 0:
        plm[l - 1] = (
            x
            * (2 * l - 1)
            * (1 if (l - 1) % 2 == 0 else -1)
            * double_fact(2 * l - 3)
            * np.power(1 - x**2, (l - 1) / 2)
        )

    # 2b) Calculate rest of the terms

    # Importantly, we will run into issues when elements of x = +-1
    # First calculate what we can and make everything else equal to 0
    # Note that 2 * x/sqrt(1 - x^2) approaches +-infinity rather slowly
    # so no need for tolerance regarding how close abs(x) is to 1
    mul_term = np.zeros_like(x)
    zero_mask = np.abs(x) == 1
    nonzero_mask = ~zero_mask
    mul_term[nonzero_mask] = -2 * x[nonzero_mask] / np.sqrt(1 - x[nonzero_mask] ** 2)

    for m in range(l - 2, -1, -1):
        r1 = mul_term * (m + 1) / ((l + m + 1) * (l - m)) * plm[m + 1]
        r2 = -1 / ((l + m + 1) * (l - m)) * plm[m + 2]
        plm[m] = r1 + r2

    # Now if we consider x = +-1, all entries will be 0 except for m = 0
    # This is due to the closed form formula having (1 - x^2) term for all m != 0
    # It is also known that at m = 0 and x = +-1:
    # A) P^0_l(1) = 1
    # B) P^0_l(-1) = (-1)^l
    set_to_negative_one = (x == -1) & (l % 2 == 1) & zero_mask
    plm[0, zero_mask] = np.where(set_to_negative_one[zero_mask], -1, 1)

    # 3) Calculate spherical harmonics Y_l^m
    idx_offset = l

    for m in range(0, l + 1):
        # Calculate for positive m
        coeff = norm_factors[l, m] * plm[m]
        q_this_m = coeff * (np.cos(m * phis) + 1j * np.sin(m * phis))

        # Entry for negative m
        if m != 0:
            # We can take advantage of the fact that Y_l^m = conjugate(Y_l^{-m}) * (-1)^m
            q_neg_m = (q_this_m * (1 if (-m) % 2 == 0 else -1)).conjugate()
            sph_y[-m + idx_offset] += q_neg_m

        # Entry for this m
        sph_y[m + idx_offset] += q_this_m

    return sph_y.T


@nb.njit
def sop_single_atom(n_b_list, l_list, unit_vecs, norm_factors):
    """
    Computes the feature vector for a single atom.

    The feature vector consists of a collection of
    Q^{N_b}_l over all N_b in n_b_list and l's in l_list.

    Args:
        n_b_list (any SORTED iterable of ints): Values of N_b
        l_list (any iterable of ints): Values of l
        unit_vecs (numpy array of size 3 arrays): Positions of atoms

    Returns:
        A numpy array of floats storing a feature vector of the
        various values of Q^{N_b}_l over N_b in n_b_list and l in l_list.
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


def calculate_all_sop(n_b_list, l_list, data):
    """
    Calculates the Steindhart Order Parameters part of the
    feature vector for each atom. The feature vector portion consists
    of a collection of Q_l^{N_b} over all N_b in n_b_list and l's in l_list.

    Args:
        n_b_list (any SORTED iterable of ints): Values of N_b.
        l_list (any iterable of ints): Values of l.
        data (OVITO data object): Information about all atoms.

    Returns:
        A numpy 2d array storing the Steindhart Order Parameters part of the
        feature vectors.
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
            [neigh.delta / np.linalg.norm(neigh.delta) for neigh in finder.find(atom)]
        )
        feature_vec.append(sop_single_atom(n_b_list, l_list, unit_vecs, norm_factors))

    return np.array(feature_vec)
