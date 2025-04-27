# Feature vector computation
N_B_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
L_LIST_FEATURES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
R_MULTS = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
SIGMA_MULT = 0.05

# Train
EPOCHS = 10
BATCH_SIZE = 512
SHUFFLE_DATASET = True
TRAIN_VAL_SPLIT = 0.8

# Coherence
ALPHA_CUTOFF = 0.196
N_B_COHERENCE = 16
L_LIST_COHERENCE = [4, 6, 8, 12]

# Outlier
PERCENT_CUTOFF = 99
