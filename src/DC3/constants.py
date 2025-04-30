"""
Global constants and configuration values for the DC3 package.
"""

import os
import numpy as np

# Locations
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "saved_pretrained_data")
SAVED_PERFECT_MD_DIR = os.path.join(
    BASE_DIR, "saved_pretrained_data", "perfect_lattices", "md_results"
)
SAVED_PERFECT_FEAT_DIR = os.path.join(
    BASE_DIR, "saved_pretrained_data", "perfect_lattices", "features"
)
SAVED_SYNTH_FEAT_DIR = os.path.join(
    BASE_DIR, "saved_pretrained_data", "synthetic_data_features"
)
SAVED_ML_MODELS_DIR = os.path.join(
    BASE_DIR, "saved_pretrained_data", "ml_info", "models"
)
SAVED_ML_STATS_DIR = os.path.join(BASE_DIR, "saved_pretrained_data", "ml_info", "stats")
SAVED_OUTLIER_DIR = os.path.join(BASE_DIR, "saved_pretrained_data", "outlier_info")
SAVED_FULL_MODEL_PATH = os.path.join(
    BASE_DIR, "saved_pretrained_data", "dc3_full_model.pth"
)

# Feature vector computation
N_B_ARR = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64)
L_ARR_FEATURES = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64
)
R_MULTS = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
SIGMA_MULT = 0.05

# Synthetic data
TEMPS = np.linspace(0.01, 0.25, 10)

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
