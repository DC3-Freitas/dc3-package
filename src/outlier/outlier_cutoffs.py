import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from constants import *

import ovito
from features.compute_all import compute_feature_vectors


def compute_ref_vec(
    ref_folder: str, means: np.ndarray, stds: np.ndarray
) -> dict[str, np.ndarray]:
    ref_vec = {}

    for f in tqdm(os.listdir(ref_folder)):
        if f.endswith(".gz"):

            pipeline = ovito.io.import_file(os.path.join(ref_folder, f))
            lattice = pipeline.compute(0)

            # Normalized target vector
            data = compute_feature_vectors(lattice)
            data = (data - means) / (stds + 1e-6)

            # Name should be in the form <strcture>_number.gz
            label = f.split(".")[0]

            ref_vec[label] = data.mean(axis=0)

    return ref_vec


def compute_delta_cutoff(
    synthetic_data_folder: str, ref_folder: str
) -> dict[str, float]:
    distances = {}
    all_data = []
    all_labels = []

    # 1) Get data
    for f in tqdm(os.listdir(synthetic_data_folder)):
        if f.endswith(".gz"):
            # All the vectors
            data = np.loadtxt(os.path.join(synthetic_data_folder, f))

            # Check data integrity
            if np.any(np.isnan(data)):
                print(f"Skipping {f} due to NaN values")
                continue

            # Name should be in the form <strcture>_number.gz
            label = f.split("_")[0]

            # Add data
            all_data.append(data)
            all_labels += [label] * data.shape[0]

    # 2) Proccess data
    all_data = np.vstack(all_data)
    all_labels = np.array(all_labels)

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # 3) Compute
    ref_vec = compute_ref_vec(ref_folder, means, stds)

    for data, label in zip(all_data, all_labels):
        normalized_data = (data - means) / (stds + 1e-6)

        # Add to distances
        if not label in distances:
            distances[label] = []

        distances[label].append(np.linalg.norm(normalized_data - ref_vec[label]))

    # 4) Cutoff
    for label in distances.keys():
        distances[label] = np.percentile(np.array(distances[label]), PERCENT_CUTOFF)

    return distances


if __name__ == "__main__":
    delta_cutoffs = compute_delta_cutoff(
        "ml_dataset/data", "lattice/lammps_lattices/data"
    )
    df = pd.DataFrame(delta_cutoffs.items(), columns=["label", "cutoff"])
    df.to_csv("delta_cutoffs.csv", index=False)
