import math
import numpy as np
import numba as nb


def precalculate_sop_norm_factors(max_l: int) -> np.ndarray:
    """
    Calculates the following part of spherical harmonics:
    sqrt[(2l + 1) / 4pi * (l - m)! / (l + m)!]
    for all l and m up to the given max_l.

    Args:
        max_l: maximum l value we will ever consider for
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
def double_fact(i: int) -> int:
    """
    Calculates i!! and returns 1 if i < 0

    Args:
        i: number to calculate double factorial of
    Returns:
        i! if i >= 0, otherwise 1
    """
    result = 1
    for x in range(i, 0, -2):
        result *= x
    return result


@nb.njit
def calc_spherical_harmonics(
    l: int, thetas: np.ndarray, phis: np.ndarray, norm_factors: np.ndarray
) -> np.ndarray:
    """
    Calculates spherical harmonics for each atom described
    by thetas and phis for m=-l...+l

    **Note: due to numba optimization, double_fact(2 * l - 1) can
    overflow if l is large. The current l=16 maximum is as large
    as it can go without overflowing!**

    Args:
        l: subscript of Y
        thetas: polar angles
        phis: azimuthal angles
    Returns:
        Numpy 2d array sph_y where sph_y[i] stores the spherical harmonic values for
        atom i (i-th neighbor) for m=-l...+l in that order.
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
