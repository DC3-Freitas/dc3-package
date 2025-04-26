import numpy as np
from features.sop import calculate_all_sop
from features.rsf import calculate_all_rsf
from constants import *


def compute_feature_vectors(data, save_path=None):
    """
    Computes feature vectors given data. Does not normalize.

    Args:
        data (OVITO data object): Information about all atoms.
        save_path (string): Where to save the information.
    Returns:
        np.array: if save_path is None, returns the feature vector; None otherwise.
    """
    # 1) Computes the feature vector
    sop_features = calculate_all_sop(N_B_LIST, L_LIST_FEATURES, data)
    rsf_features = calculate_all_rsf(N_B_LIST, R_MULTS, SIGMA_MULT, data)
    feature_vector = np.hstack((sop_features, rsf_features))

    # 2) Save
    if save_path is None:
        return feature_vector
    np.savetxt(save_path, feature_vector)