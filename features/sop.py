import numpy as np
import numba as nb
from ovito.data import NearestNeighborFinder
from tqdm import tqdm


@nb.njit
def fact(i):
    """
    Calculates i! and returns 1 is i < 0
    Args:
        i (int): Number to calculate factorial of
    Returns:
        i! if i >= 0, otherwise 1
    """
    result = 1
    for x in range(2, i + 1):
        result *= x
    return result


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
def calc_spherical_harmonics(l, theta, phi):
    """
    Calculates spherical harmonics for m=-l...l given l,
    theta, and phi.

    Args:
        l (float): Subscript of Y
        theta (float): Polar angle
        phi (float): Azimuthal angle
    Returns:
        Numpy array of length 2l+1 storing spherical harmonics
        for m=-l...l in that order.
    """
    # 1) Variables
    sph_y = np.zeros(l * 2 + 1, dtype=np.complex128)
    plm = np.zeros(l + 1)

    # 2) Calculate associated legendre polynomial P_l^m
    # We use the reccurence: P_l^m = something in terms of P^{m+1}_l and P^{m+2}_l
    # Note: m = -l...l but only need to calculate 0...l due to properties

    # 2a) Base cases
    x = np.cos(theta)

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
    mul_term = -2 * x / np.sqrt(1 - x**2)

    for m in range(l - 2, -1, -1):
        r1 = mul_term * (m + 1) / ((l + m + 1) * (l - m)) * plm[m + 1]
        r2 = -1 / ((l + m + 1) * (l - m)) * plm[m + 2]
        plm[m] = r1 + r2

    # 3) Calculate spherical harmonics Y_l^m
    idx_offset = l

    for m in range(0, l + 1):
        # Calculate for positive m
        coeff = np.sqrt((2 * l + 1) / (4 * np.pi) * fact(l - m) / fact(l + m)) * plm[m]
        q_this_m = coeff * (np.cos(m * phi) + 1j * np.sin(m * phi))

        # Entry for negative m
        if m != 0:
            # We can take advantage of the fact that Y_l^m = conjugate(Y_l^{-m}) * (-1)^m
            q_neg_m = (q_this_m * (1 if (-m) % 2 == 0 else -1)).conjugate()
            sph_y[-m + idx_offset] += q_neg_m

        # Entry for this m
        sph_y[m + idx_offset] += q_this_m

    return sph_y


@nb.njit
def sop_single_atom(N_b_list, l_list, unit_vecs):
    """
    Computes the feature vector for a single atom.

    The feature vector consists of a collection of
    Q^{N_b}_l over all N_b in N_b_list and l's in l_list.

    Args:
        N_b_list (any SORTED iterable of ints): Values of N_b.
        l_list (any iterable of ints): Values of l.
        unit_vecs (numpy array of size 3 arrays): Positions of atoms

    Returns:
        A numpy array of floats storing a feature vector of the
        various values of Q^{N_b}_l over N_b in N_b_list and l in l_list.
    """
    # 1) Extract unit vectors and convert to spherical (neighbors guarenteed to be in sorted order)
    x_coords, y_coords, z_coords = unit_vecs[:, 0], unit_vecs[:, 1], unit_vecs[:, 2]
    thetas = np.arccos(z_coords)
    phis = np.arctan2(y_coords, x_coords)

    # 2) Calculate each Q_l^{N_b}
    Q = []

    for l in l_list:
        curAtom = 0

        # A tricky math thing is that we need to keep track of all different little q at once
        # This is since norm(sum of vectors) is not the same as sum(norm of vectors)
        q_accum_all = np.zeros(2 * l + 1, dtype=np.complex128)

        for N_b in N_b_list:
            while curAtom < N_b:
                q_accum_all += calc_spherical_harmonics(
                    l, thetas[curAtom], phis[curAtom]
                )
                curAtom += 1

            Q.append(
                np.sqrt(
                    (4 * np.pi) / (2 * l + 1) * (np.linalg.norm(q_accum_all / N_b) ** 2)
                )
            )

    # 3) Rearrange elements so that its flattened version of shape=(N_b, l)
    return np.array(Q).reshape((len(l_list), len(N_b_list))).T.flatten()


def calculate_all_sop(N_b_list, l_list, data):
    """
    Calculates the Steindhart Order Parameters part of the
    feature vector for each atom. The feature vector portion consists
    of a collection of Q_l^{N_b} over all N_b in N_b_list and l's in l_list.

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
    for atom in tqdm(range(num_atoms), desc="SOP: Calculating"):
        unit_vecs = np.array(
            [neigh.delta / np.linalg.norm(neigh.delta) for neigh in finder.find(atom)]
        )
        feature_vec.append(sop_single_atom(N_b_list, l_list, unit_vecs))

    return np.array(feature_vec)
