import numpy as np
import json
import os
from DC3.constants import PERCENT_CUTOFF, SAVED_SYNTH_FEAT_DIR, SAVED_PERFECT_FEAT_DIR

def compute_ref_vec(
    data: list[tuple[str, np.ndarray]], means: np.ndarray, stds: np.ndarray, save_dir: str | None = None
) -> dict[str, np.ndarray]:
    """
    TODO
    """
    ref_vec = {}

    for structure, features in data:
        normalized_features = (features - means) / (stds + 1e-6)
        ref_vec[structure] = normalized_features.mean(axis=0)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, "ref_vecs.npz"), **ref_vec)

    return ref_vec


def compute_delta_cutoff(
    data: list[tuple[str, np.ndarray]],
    ref_vecs: dict[str, np.ndarray],
    means: np.ndarray,
    stds: np.ndarray,
    save_dir: str | None = None
) -> dict[str, float]:
    """ 
    TODO
    """

    # Get distances
    distances = {}

    for structure, features in data:
        normalized_features = (features - means) / (stds + 1e-6)

        # Add to distances
        if not structure in distances:
            distances[structure] = []

        distances[structure].extend(
            np.linalg.norm(
                normalized_features - ref_vecs[structure], axis=1
            ).tolist()
        )
        
    # Calculate 99-th percentile
    for label in distances.keys():
        distances[label] = np.percentile(np.array(distances[label]), PERCENT_CUTOFF)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "label_map.json"), "w") as f:
            json.dump(distances, f)

    return distances
