"""
Data handling and preparation for crystal structure classification.
Manages loading and preprocessing of perfect and synthetic lattice data
for training and evaluating the DC3 model.
"""

import os
import numpy as np
from ovito.io import import_file
from DC3.compute_features.compute_all import compute_feature_vectors
from DC3.constants import SAVED_PERFECT_FEAT_DIR, SAVED_SYNTH_FEAT_DIR
from DC3.ml_dataset.process_lattices import generate_from_perfect_lattices


class DataHandler:
    """
    Loads and prepares perfect and synthetic lattice data for model training and evaluation.
    """

    def __init__(self, structure_map: dict[str, str | None]) -> None:
        """
        Initializes the DataHandler such that it supports storing perfect
        and synthetic data.

        Args:
            structure_map: mapping from structure name to either a LAMMPS file path
                           (for generating features) or None (to load precomputed features)
        """
        self.perfect_lattice_data = []
        self.synthetic_data = []

        to_gen_paths = []
        to_gen_structures = []

        for structure, path in structure_map.items():
            # Path specified: read and generate
            if path is not None:
                perfect_lattice = import_file(path).compute(0)
                self.perfect_lattice_data.append(
                    (structure, compute_feature_vectors(perfect_lattice))
                )

                to_gen_paths.append(path)
                to_gen_structures.append(structure)

            # Path not specified: use existing
            else:
                structure_perfect_dir = os.path.join(
                    SAVED_PERFECT_FEAT_DIR, f"{structure}.npy"
                )
                structure_synth_dir = os.path.join(SAVED_SYNTH_FEAT_DIR, f"{structure}")

                assert os.path.exists(
                    structure_perfect_dir
                ), f"Perfect lattices for {structure} must exist"
                assert os.path.exists(
                    structure_synth_dir
                ), f"Synthetic data for {structure} must exist"

                features = np.load(structure_perfect_dir)
                self.perfect_lattice_data.append((structure, features))

                for f in os.listdir(structure_synth_dir):
                    features = np.load(os.path.join(structure_synth_dir, f))
                    self.synthetic_data.append((structure, features))

        # Generate all synthetic data at once (in case we want to parallelize things)
        if len(to_gen_paths) > 0:
            self.synthetic_data.extend(
                generate_from_perfect_lattices(to_gen_paths, to_gen_structures)
            )

    def get_perfect_lattice_data(self) -> list[tuple[str, np.ndarray]]:
        """Defensively returns all perfect (label, features)"""
        return [
            (structure, np.copy(features))
            for structure, features in self.perfect_lattice_data
        ]

    def get_synthetic_data(self) -> list[tuple[str, np.ndarray]]:
        """Defensively returns all synthetic (label, features)"""
        return [
            (structure, np.copy(features))
            for structure, features in self.synthetic_data
        ]
