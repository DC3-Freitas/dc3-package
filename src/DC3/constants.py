import os
import numpy as np

# Folders
BASE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
PERFECT_LATTICE_MD_RESULTS_DIRECTORY = os.path.join(BASE_DIRECTORY, "saved_data", "perfect_lattices", "md_results")
PERFECT_LATTICE_FEATURES_DIRECTORY = os.path.join(BASE_DIRECTORY, "saved_data", "perfect_lattices", "features")
SYNTHETIC_DATA_FEATURES_DIRECTORY = os.path.join(BASE_DIRECTORY, "saved_data", "synthetic_data_features")

# Feature vector computation
N_B_ARR = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64)
L_ARR_FEATURES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64)
R_MULTS = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
SIGMA_MULT = 0.05

# Train
EPOCHS = 10
BATCH_SIZE = 512
SHUFFLE_DATASET = True
TRAIN_VAL_SPLIT = 0.8

# Coherence
ALPHA_CUTOFF = 0.196
N_B_COHERENCE = 16
L_ARR_COHERENCE = np.array([4, 6, 8, 12], dtype=np.int64)

# Outlier
PERCENT_CUTOFF = 99
