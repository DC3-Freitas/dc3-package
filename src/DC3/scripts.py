import os
import numpy as np
from ovito.io import import_file
from DC3.compute_features.compute_all import compute_feature_vectors
from DC3.ml_dataset.process_lattices import generate_from_perfect_lattices
from DC3.ml_dataset.dataset import CrystalDataset
from DC3.ml.model import MLPModel
from DC3.ml.train import train
from DC3.outlier.outlier_cutoffs import compute_ref_vec, compute_delta_cutoff
from DC3.constants import (
    SAVED_DIR,
    SAVED_PERFECT_MD_DIR,
    SAVED_PERFECT_FEAT_DIR,
    SAVED_SYNTH_FEAT_DIR,
    SAVED_ML_MODELS_DIR,
    SAVED_ML_STATS_DIR,
    SAVED_OUTLIER_DIR,
)
from DC3.dc3 import create_model


def gen_features_from_perfect_lattices():
    os.makedirs(SAVED_PERFECT_FEAT_DIR, exist_ok=True)

    for f in os.listdir(SAVED_PERFECT_MD_DIR):
        data = import_file(os.path.join(SAVED_PERFECT_MD_DIR, f)).compute(0)
        # Expects f to be in the form <structure_name>.gz and the suffix is replaced with .npy
        np.save(
            os.path.join(SAVED_PERFECT_FEAT_DIR, f"{f[:-3]}.npy"),
            compute_feature_vectors(data),
        )


def gen_synthetic_features():
    lattice_paths = []
    save_folders = []

    for f in os.listdir(SAVED_PERFECT_MD_DIR):
        lattice_paths.append(os.path.join(SAVED_PERFECT_MD_DIR, f))

        # Expect f to be in the form <structure_name>.gz
        save_folders.append(os.path.join(SAVED_SYNTH_FEAT_DIR, f.split(".")[0]))

    generate_from_perfect_lattices(lattice_paths, None, save_folders)


def calculate_label_map_and_train_model():
    # Create dataset
    data = []
    for structure in os.listdir(SAVED_SYNTH_FEAT_DIR):
        for f in os.listdir(os.path.join(SAVED_SYNTH_FEAT_DIR, structure)):
            features = np.load(os.path.join(SAVED_SYNTH_FEAT_DIR, structure, f))
            data.append((structure, features))

    # Train and save model
    dataset = CrystalDataset(data, SAVED_DIR)
    model = MLPModel(len(dataset.label_map), dataset.means, dataset.stds)
    train(model, dataset, "pretrained_model", SAVED_ML_MODELS_DIR, SAVED_ML_STATS_DIR)


def calculate_outliers():
    # Get all data
    perfect_data = []
    synthetic_data = []

    for f in os.listdir(SAVED_PERFECT_FEAT_DIR):
        features = np.load(os.path.join(SAVED_PERFECT_FEAT_DIR, f))
        # Expects form <structure_name>.npy
        perfect_data.append((f[:-4], features))

    for structure in os.listdir(SAVED_SYNTH_FEAT_DIR):
        for f in os.listdir(os.path.join(SAVED_SYNTH_FEAT_DIR, structure)):
            features = np.load(os.path.join(SAVED_SYNTH_FEAT_DIR, structure, f))
            synthetic_data.append((structure, features))

    # Create dataset for means and averages
    dataset = CrystalDataset(synthetic_data)

    # Calculate here
    ref_vecs = compute_ref_vec(
        perfect_data, dataset.means, dataset.stds, SAVED_OUTLIER_DIR
    )
    compute_delta_cutoff(
        synthetic_data, ref_vecs, dataset.means, dataset.stds, SAVED_OUTLIER_DIR
    )


def create_and_save_dc3():
    # Note that the ml model here may be different than whatever model is currently saved
    structure_map = {}
    for f in os.listdir(SAVED_PERFECT_FEAT_DIR):
        # Expects form <structure_name>.npy
        structure_map[f[:-4]] = None
    
    create_model(structure_map).save("dc3_full_model", SAVED_DIR)


def main():
    gen_features_from_perfect_lattices()
    gen_synthetic_features()
    calculate_label_map_and_train_model()
    calculate_outliers()
    create_and_save_dc3()


if __name__ == "__main__":
    main()
