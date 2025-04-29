from ovito.io import import_file
from DC3.compute_features.compute_all import compute_feature_vectors
from DC3.ml_dataset.process_lattices import generate_from_perfect_lattices
import numpy as np
import os
from DC3.constants import SAVED_PERFECT_FEAT_DIR, SAVED_SYNTH_FEAT_DIR

class DataHandler:
    def __init__(self, structure_map: dict[str, str | None]) -> None:
        """
        TODO
        """
        self.perfect_lattice_data = []
        self.synthetic_data = []

        to_gen_paths = []
        to_gen_structures = []

        for structure, path in structure_map.items():
            if path is not None:
                perfect_lattice = import_file(path).compute(0)
                self.perfect_lattice_data.append((structure, compute_feature_vectors(perfect_lattice)))

                to_gen_paths.append(path)
                to_gen_structures.append(structure)
            else:
                structure_perfect_dir = os.path.join(SAVED_PERFECT_FEAT_DIR, f"{structure}.npy")
                structure_synth_dir = os.path.join(SAVED_SYNTH_FEAT_DIR, f"{structure}")

                assert os.path.exists(structure_perfect_dir), f"Perfect lattices for {structure} must exist"
                assert os.path.exists(structure_synth_dir), f"Synthetic data for {structure} must exist"

                features = np.load(structure_perfect_dir)
                self.perfect_lattice_data.append((structure, features))
                
                for f in os.listdir(structure_synth_dir):
                    features = np.load(os.path.join(structure_synth_dir, f))
                    self.synthetic_data.append((structure, features))

        # Generate all synthetic data at once
        if len(to_gen_paths) > 0:
            self.synthetic_data.extend(generate_from_perfect_lattices(to_gen_paths, to_gen_structures))

    def get_perfect_lattice_data(self) -> list[tuple[str, np.ndarray]]:
        """Defensively returns all perfect (label, features)"""
        return [(structure, np.copy(features)) for structure, features in self.perfect_lattice_data]
    
    def get_synthetic_data(self) -> list[tuple[str, np.ndarray]]:
        """Defensively returns all synthetic (label, features)"""
        return [(structure, np.copy(features)) for structure, features in self.synthetic_data]