import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from constants import *


def compute_ref_vec(
    ref_folder: str, means: np.ndarray, stds: np.ndarray
) -> dict[str, np.ndarray]:
    """
    TODO
    """
    ref_vec = {}

    for f in os.listdir(ref_folder):
        if f.endswith(".npy"):
            # Load and normalize
            data = np.load(os.path.join(ref_folder, f))
            normalized_data = (data - means) / (stds + 1e-6)

            # Name should be in the form <strcture>.npy
            ref_vec[f.split(".")[0]] = normalized_data.mean(axis=0)

    return ref_vec


def compute_delta_cutoff(
    synthetic_data_folder: str, ref_vecs: dict[str, np.ndarray], means: np.ndarray, stds: np.ndarray
) -> dict[str, float]:
    """
    """

    # Get distances
    distances = {}

    for structure_name in os.listdir(synthetic_data_folder):
        if os.path.isdir(os.path.join(synthetic_data_folder, structure_name)):
            for f in os.listdir(os.path.join(synthetic_data_folder, structure_name)):
                if f.endswith(".npy"):
                    data = np.load(os.path.join(synthetic_data_folder, structure_name, f))
                    normalized_data = (data - means) / (stds + 1e-6)

                    # Add to distances
                    if not structure_name in distances:
                        distances[structure_name] = []

                    distances[structure_name].extend(np.linalg.norm(normalized_data - ref_vecs[structure_name], axis=0).tolist())

    # Calculate 99-th percentile
    for label in distances.keys():
        distances[label] = np.percentile(np.array(distances[label]), PERCENT_CUTOFF)

    return distances
