from compute_features.compute_all import compute_feature_vectors
from ovito.io import import_file
import numpy as np
import os

if __name__ == "__main__":
    for f in os.listdir("lattice/md_results"):
        data = import_file(os.path.join("lattice/md_results", f)).compute(0)
        np.save(f"lattice/features/{f.split(".")[0]}.npy", compute_feature_vectors(data))