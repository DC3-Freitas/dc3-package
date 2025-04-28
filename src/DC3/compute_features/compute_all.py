import numpy as np
from ovito.data import DataCollection
from DC3.compute_features.sop import calculate_all_sop
from DC3.compute_features.rsf import calculate_all_rsf
from DC3.constants import N_B_ARR, L_ARR_FEATURES, R_MULTS, SIGMA_MULT


def compute_feature_vectors(data: DataCollection) -> np.ndarray:
    """
    Computes feature vectors given data. Does not normalize.

    Args:
        data: information about all atoms.
    Returns:
        The feature vector representing the data
    """
    # 1) Computes the feature vector
    sop_features = calculate_all_sop(N_B_ARR, L_ARR_FEATURES, data)
    rsf_features = calculate_all_rsf(N_B_ARR, R_MULTS, SIGMA_MULT, data)
    feature_vector = np.hstack((sop_features, rsf_features))

    # 2) Return
    return feature_vector
