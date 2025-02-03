import numpy as np
from sop import calculate_all_sop
from rsf import calculate_all_rsf

N_B_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
L_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
R_MULTS = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
SIGMA_MULT = 0.05


def compute_feature_vectors(data, save_path):
    """
    Computes feature vectors given data, normalizes so that
    each component has mean 0 and std 1, and writes them
    to path specified by save_path.

    Args:
        data (OVITO data object): Information about all atoms.
        save_path (string): Where to save the information.
    Returns:
        Nothing
    """
    # 1) Computes the feature vector
    sop_features = calculate_all_sop(N_B_LIST, L_LIST, data)
    rsf_features = calculate_all_rsf(N_B_LIST, R_MULTS, SIGMA_MULT, data)
    feature_vector = np.hstack((sop_features, rsf_features))

    # 2) Normalize feature fector
    means = np.mean(feature_vector, axis=0)
    stds = np.std(feature_vector, axis=0)
    feature_vector = (feature_vector - means) / stds

    # 3) Save
    np.savetxt(save_path, feature_vector)
